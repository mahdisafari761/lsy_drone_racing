"""Filter a learned racing actor using an independent nominal dynamics model.

The internal simulator only receives public observations and nominal parameters.
It never receives data or parameters from the race simulator.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING, Any

import jax
import jax.numpy as jnp
import numpy as np
from crazyflow import Sim
from crazyflow.sim.functional import attitude_control, state_control
from jax.scipy.spatial.transform import Rotation

from lsy_drone_racing.control._attitude_state import invert_geometry
from lsy_drone_racing.control._fast_policy_jax import actor, basic_state, features
from lsy_drone_racing.control.level2_fast import Level2Fast

if TYPE_CHECKING:
    from ml_collections import ConfigDict

_ANGLES = np.arange(8) * np.pi / 4
_OFFSETS = np.zeros((19, 4), dtype=np.float32)
_OFFSETS[1:9, :2] = np.c_[np.cos(_ANGLES), np.sin(_ANGLES)] * 0.15
_OFFSETS[9:17, :2] = np.c_[np.cos(_ANGLES), np.sin(_ANGLES)] * 0.30
_OFFSETS[17:, 3] = [-0.15, 0.15]


@lru_cache(maxsize=4)
def build_predictor(
    drone: str,
    dynamics: str,
    sim_freq: int,
    attitude_freq: int,
    freq: int,
    agile: bool,
    unseen_limit: float,
) -> tuple:
    """Compile a model shared across races without sharing their dynamic state."""
    samples = _OFFSETS
    count = len(samples)
    horizon = 20
    sim = Sim(
        n_worlds=count,
        drone=drone,
        dynamics=dynamics,
        control="state" if agile else "attitude",
        state_freq=freq,
        freq=sim_freq,
        attitude_freq=attitude_freq,
    )
    state_params = sim.data.controls.state.params if agile else None
    step_sim = sim.build_step_fn()
    offsets = jnp.asarray(samples)
    scale = jnp.array([jnp.pi / 2] * 3 + [0.3])
    bias = jnp.array([0.0, 0.0, 0.0, 0.5])
    frame_centers = jnp.array([[0, 0, 0.28], [0, 0, -0.28], [0, -0.28, 0], [0, 0.28, 0]])
    frame_half = jnp.array(
        [[0.01, 0.36, 0.08], [0.01, 0.36, 0.08], [0.01, 0.08, 0.36], [0.01, 0.08, 0.36]]
    )

    @jax.jit
    def predict(
        data: Any,
        public: dict,
        history: jax.Array,
        previous: jax.Array,
        parameters: dict,
        mass: jax.Array,
    ) -> tuple:
        tiled = jax.tree.map(lambda x: jnp.broadcast_to(x, (count,) + x.shape), public)
        states = data.states.replace(
            **{k: tiled[k][:, None, :] for k in ["pos", "vel", "quat", "ang_vel"]}
        )
        data = data.replace(
            states=states, params=data.params.replace(mass=jnp.full_like(data.params.mass, mass))
        )
        history = jnp.broadcast_to(history, (count, 2, 16))
        previous = jnp.broadcast_to(previous, (count, 2, 4))
        gate_rot = Rotation.from_quat(public["gates_quat"]).as_matrix()
        batch = jnp.arange(count)
        first_data = data
        first_raw = jnp.zeros((count, 4))

        def scan(carry: tuple, index: jax.Array) -> tuple:
            data, obs, history, previous, minimum, first_data, first_raw = carry
            raw = actor(parameters, features(obs, history, previous))
            raw = raw + offsets * jnp.maximum(0.0, 1.0 - index / 10.0)
            command = jnp.clip(raw, -1, 1) * scale + bias
            before = data.states.pos[:, 0]
            if agile:
                desired = Rotation.from_euler("xyz", command[:, :3]).as_matrix()
                current = Rotation.from_quat(data.states.quat[:, 0]).as_matrix()
                rates = offsets[:, :3] * (jnp.pi / 2) * 6.0 * jnp.maximum(0.0, 1.0 - index / 10.0)
                rates = rates * (obs["n_gates_passed"] == 0)[:, None]
                full = invert_geometry(
                    data.states.pos[:, 0],
                    data.states.vel[:, 0],
                    desired,
                    current,
                    command[:, 3],
                    rates,
                    state_params,
                    jnp,
                )
                data = state_control(data, full[:, None, :])
            else:
                data = attitude_control(data, command[:, None, :])
            data = step_sim(data, sim_freq // freq)
            first_data = jax.lax.cond(index == 0, lambda _: data, lambda _: first_data, None)
            first_raw = jnp.where(index == 0, raw, first_raw)
            position = data.states.pos[:, 0]
            local = jnp.einsum(
                "gji,ngj->ngi", gate_rot, position[:, None, :] - public["gates_pos"][None, :, :]
            )
            drone_rot = Rotation.from_quat(data.states.quat[:, 0]).as_matrix()
            relative_rot = jnp.einsum("gji,njk->ngik", gate_rot, drone_rot)
            drone_half = jnp.array([0.07, 0.07, 0.02])
            projected = jnp.einsum("ngij,j->ngi", jnp.abs(relative_rot), drone_half)
            gap = (
                jnp.abs(local[:, :, None, :] - frame_centers[None, None, :, :])
                - frame_half[None, None, :, :]
                - projected[:, :, None, :]
            )
            signed = jnp.linalg.norm(jnp.maximum(gap, 0.0), axis=-1) + jnp.minimum(
                jnp.max(gap, axis=-1), 0.0
            )
            frame_distance = jnp.min(signed, axis=(1, 2))
            world_half = jnp.einsum("nij,j->ni", jnp.abs(drone_rot), drone_half)
            pole_gap = (
                jnp.abs(position[:, None, :2] - public["obstacles_pos"][None, :, :2])
                - world_half[:, None, :2]
            )
            pole_distance = (
                jnp.linalg.norm(jnp.maximum(pole_gap, 0.0), axis=-1)
                + jnp.minimum(jnp.max(pole_gap, axis=-1), 0.0)
                - 0.015
            )
            pole_distance = jnp.where(
                position[:, None, 2] - world_half[:, None, 2] < public["obstacles_pos"][None, :, 2],
                pole_distance,
                10.0,
            )
            clearance = jnp.minimum(frame_distance, jnp.min(pole_distance, axis=-1))
            clearance = jnp.where(
                obs["n_gates_passed"] < obs["gate_sequence"].shape[-1], clearance, 10.0
            )
            if unseen_limit > 0:
                first = public["gate_sequence"][0]
                delta = position - public["gates_pos"][first]
                approaching = (delta @ gate_rot[first, :, 0]) * public["gate_sequence_direction"][
                    0
                ] < 0
                unknown = (
                    (~public["gates_visited"][first])
                    & (obs["n_gates_passed"] == 0)
                    & approaching
                    & (jnp.linalg.norm(delta[:, :2], axis=-1) < 1.3)
                    & (jnp.linalg.norm(delta[:, :2], axis=-1) > 0.7)
                )
                excess = jnp.where(
                    unknown,
                    jnp.maximum(
                        jnp.linalg.norm(data.states.vel[:, 0, :2], axis=-1) - unseen_limit, 0.0
                    ),
                    0.0,
                )
                clearance = jnp.minimum(
                    clearance, jnp.where(excess > 0, 0.01 - 0.02 * excess, 10.0)
                )
            minimum = jnp.minimum(minimum, clearance)
            passed = obs["n_gates_passed"]
            sequence_index = jnp.minimum(passed, obs["gate_sequence"].shape[-1] - 1)
            gate = obs["gate_sequence"][batch, sequence_index]
            direction = obs["gate_sequence_direction"][batch, sequence_index]
            current_rot = gate_rot[gate]
            center = public["gates_pos"][gate]
            start_local = jnp.einsum("nji,nj->ni", current_rot, before - center)
            end_local = local[batch, gate]
            fraction = jnp.clip(
                -start_local[:, 0] / (end_local[:, 0] - start_local[:, 0] + 1e-8), 0.0, 1.0
            )
            intersection = start_local[:, 1:] + fraction[:, None] * (
                end_local[:, 1:] - start_local[:, 1:]
            )
            crossed = (
                (start_local[:, 0] * direction <= 0)
                & (end_local[:, 0] * direction > 0)
                & jnp.all(jnp.abs(intersection) < 0.2, axis=-1)
            )
            new_obs = dict(obs)
            new_obs.update(
                {k: getattr(data.states, k)[:, 0] for k in ["pos", "vel", "quat", "ang_vel"]}
            )
            new_obs["n_gates_passed"] = jnp.minimum(
                passed + crossed, obs["gate_sequence"].shape[-1]
            )
            history = jnp.concatenate([history[:, 1:, :], basic_state(obs)[:, None, :]], axis=1)
            previous = jnp.concatenate([previous[:, 1:, :], raw[:, None, :]], axis=1)
            return (data, new_obs, history, previous, minimum, first_data, first_raw), (
                new_obs["n_gates_passed"],
                clearance,
            )

        carry, traces = jax.lax.scan(
            scan,
            (data, tiled, history, previous, jnp.full(count, 10.0), first_data, first_raw),
            jnp.arange(horizon),
        )
        progress_trace, _ = traces
        _, end_obs, _, _, clearance, first_data, first_raw = carry
        violation = jnp.maximum(0.01 - clearance, 0.0)
        risk = violation**2
        passed = end_obs["n_gates_passed"]
        target_index = jnp.minimum(passed, public["gate_sequence"].shape[0] - 1)
        target = public["gates_pos"][public["gate_sequence"][target_index]]
        distance = jnp.linalg.norm(end_obs["pos"] - target, axis=-1)
        distance = jnp.where(passed >= public["gate_sequence"].shape[0], 0.0, distance)
        cost = (
            1e5 * risk
            + jnp.sum(offsets**2, axis=-1)
            + 0.04 * distance
            - 0.1 * passed
            - 0.05 * jnp.sum(progress_trace, axis=0)
        )
        selected = jnp.where(clearance[0] >= 0.01, 0, jnp.argmin(cost))

        def select(x: jax.Array) -> jax.Array:
            return (
                jnp.broadcast_to(x[selected][None], x.shape)
                if x.ndim > 0 and x.shape[0] == count
                else x
            )

        chosen_data = jax.tree.map(select, first_data)
        return chosen_data, first_raw[selected], clearance[0], selected, clearance[selected]

    return sim.data, predict


class Level2Predictive85(Level2Fast):
    """Use a short physical preview to correct predicted collisions."""

    POLICY_FILE = "level2_predictive_85.npz"
    AGILE = False
    UNSEEN_LIMIT = 0.0

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Initialize a private nominal model from public configuration."""
        super().__init__(obs, info, config)
        self.model_data, self.predict = build_predictor(
            config.sim.drone,
            config.sim.dynamics,
            config.sim.freq,
            config.sim.attitude_freq,
            config.env.freq,
            self.AGILE,
            self.UNSEEN_LIMIT,
        )
        self.parameters = {"actor": [(jnp.asarray(w), jnp.asarray(b)) for w, b in self.layers]}
        for name in ["residual", "phase_residual", "mlp"]:
            value = getattr(self, name)
            if value is not None:
                self.parameters[name] = jax.tree.map(jnp.asarray, value)
        if self.initial_layers is not None:
            self.parameters["initial_actor"] = [
                (jnp.asarray(w), jnp.asarray(b)) for w, b in self.initial_layers
            ]
            self.parameters["initial_residual"] = jax.tree.map(jnp.asarray, self.initial_residual)
        if self.mlp_phase is not None:
            self.parameters["mlp_phase"] = jnp.asarray(float(self.mlp_phase))
        self.estimated_mass = 0.0434
        self.last_velocity = None
        self.last_command = None
        self.frequency = config.env.freq
        self.interventions = 0
        # Compile during initialization; ordinary control calls retain the 20 ms budget.
        self.predict(
            self.model_data,
            {k: jnp.asarray(v) for k, v in obs.items()},
            jnp.asarray(self.history),
            jnp.zeros((2, 4)),
            self.parameters,
            jnp.asarray(self.estimated_mass),
        )[1].block_until_ready()

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Select a nearby policy command when its preview predicts a collision."""
        self.features(obs)
        if self.last_velocity is not None and self.tick > 8 and obs["pos"][2] > 0.12:
            acceleration = (obs["vel"][2] - self.last_velocity[2]) * self.frequency + 9.81
            if acceleration > 3.0:
                qx, qy = np.asarray(obs["quat"])[:2]
                zz = 1.0 - 2.0 * (qx * qx + qy * qy)
                estimate = self.last_command[3] * zz / acceleration
                self.estimated_mass = 0.95 * self.estimated_mass + 0.05 * np.clip(
                    estimate, 0.0384, 0.0484
                )
        previous = jnp.asarray(np.stack([self.previous_action, self.pending_action]))
        model_data, raw, clearance, selected, selected_clearance = self.predict(
            self.model_data,
            {k: jnp.asarray(v) for k, v in obs.items()},
            jnp.asarray(self.history),
            previous,
            self.parameters,
            jnp.asarray(self.estimated_mass),
        )
        self.model_data = model_data
        self.predicted_position = np.asarray(model_data.states.pos[0, 0])
        self.predicted_velocity = np.asarray(model_data.states.vel[0, 0])
        self.predicted_clearance = float(clearance)
        self.selected_clearance = float(selected_clearance)
        self.selected_correction = int(selected)
        raw = np.asarray(raw)
        self.previous_action = self.pending_action.copy()
        self.pending_action = raw.copy()
        command = self.action_low + (np.clip(raw, -1, 1) + 1) * 0.5 * (
            self.action_high - self.action_low
        )
        self.last_velocity = np.asarray(obs["vel"]).copy()
        self.last_command = command.copy()
        self.interventions += int(selected) > 0
        self.tick += 1
        return command.astype(np.float32)
