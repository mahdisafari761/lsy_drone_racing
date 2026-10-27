"""Optimize a frozen reference controller in complete, randomized Level 2 races."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.spatial.transform import Rotation

from lsy_drone_racing.control._reference_policy_jax import actor, load_policy, reference_features
from lsy_drone_racing.envs.drone_race import VecDroneRaceEnv
from lsy_drone_racing.envs.race_core import build_action_space
from lsy_drone_racing.envs.race_core import obs as public_observation
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[1]


def build_rollout(
    population: int,
    trials: int,
    steps: int = 260,
    time_scaling: bool = False,
    lead_control: bool = False,
    aperture_control: bool = False,
) -> tuple:
    """Compile the unchanged race and public actor, grouping worlds by candidate."""
    count = population * trials
    config = load_config(ROOT / "config/level2.toml")
    config.sim.render = False
    env = VecDroneRaceEnv(
        num_envs=count,
        freq=config.env.freq,
        sim_config=config.sim,
        track=config.env.track,
        sensor_range=config.env.sensor_range,
        control_mode="attitude",
        disturbances=config.env.disturbances,
        randomizations=config.env.randomizations,
        seed=1,
    )
    env.settings = env.settings.replace(autoreset=False)
    step_env = env.build_step_fn()
    params = load_policy(ROOT / "lsy_drone_racing/control/policies/level2_reference.npz")
    space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
    low, high = float(space.low[3]), float(space.high[3])
    indices = jnp.tile(jnp.arange(trials), population)
    batch = jnp.arange(count)

    @jax.jit
    def rollout(gains: jax.Array, seed: jax.Array, parameters: dict | None = None) -> tuple:
        policy = params if parameters is None else parameters
        data, _ = env._reset(env.data, seed=seed)
        data = jax.tree.map(
            lambda x: x[indices] if hasattr(x, "ndim") and x.ndim and x.shape[0] == count else x,
            data,
        )
        obs = {k: v[:, 0] for k, v in public_observation(data).items()}
        gains = jnp.repeat(gains, trials, axis=0)
        previous = jnp.zeros((count, 4))
        last_thrust = jnp.full(count, 0.0434 * 9.81)
        completed = jnp.zeros(count, bool)
        times = jnp.full(count, steps / config.env.freq)
        passed = jnp.zeros(count, jnp.int32)

        def step(carry: tuple, tick: jax.Array) -> tuple:
            data, obs, previous, last_thrust, completed, times, passed = carry
            phase = jnp.minimum(obs["n_gates_passed"], 4)
            settings = gains[batch, phase]
            encoded = reference_features(obs, previous)
            if aperture_control:
                yaw = jnp.arctan2(
                    2
                    * (
                        obs["quat"][:, 3] * obs["quat"][:, 2]
                        + obs["quat"][:, 0] * obs["quat"][:, 1]
                    ),
                    1 - 2 * (obs["quat"][:, 1] ** 2 + obs["quat"][:, 2] ** 2),
                )
                cy, sy = jnp.cos(yaw), jnp.sin(yaw)
                for advance, start in [(0, 15), (1, 25)]:
                    slot = jnp.minimum(phase + advance, 4)
                    gid = obs["gate_sequence"][batch, slot]
                    rotation = Rotation.from_quat(obs["gates_quat"][batch, gid]).as_matrix()
                    setting = gains[batch, slot]
                    offset = (
                        rotation[:, :, 1] * setting[:, 2, None]
                        + rotation[:, :, 2] * setting[:, 3, None]
                    )
                    shift = jnp.stack(
                        [
                            cy * offset[:, 0] + sy * offset[:, 1],
                            -sy * offset[:, 0] + cy * offset[:, 1],
                            offset[:, 2],
                        ],
                        axis=-1,
                    )
                    shift = jnp.where((phase + advance < 5)[:, None], shift, 0.0)
                    encoded = encoded.at[:, start : start + 3].add(shift)
            if time_scaling:
                encoded = encoded.at[:, :6].divide(settings[:, 1, None])
                current = Rotation.from_quat(obs["quat"]).as_matrix()
                yaw = jnp.arctan2(current[:, 1, 0], current[:, 0, 0])
                axis = current[:, :, 2] * last_thrust[:, None]
                axis = axis.at[:, 2].add((settings[:, 1] ** 2 - 1) * 0.0434 * 9.81)
                axis /= jnp.maximum(jnp.linalg.norm(axis, axis=-1, keepdims=True), 1e-6)
                roll = jnp.arcsin(
                    jnp.clip(axis[:, 0] * jnp.sin(yaw) - axis[:, 1] * jnp.cos(yaw), -1.0, 1.0)
                )
                pitch = jnp.arctan2(
                    axis[:, 0] * jnp.cos(yaw) + axis[:, 1] * jnp.sin(yaw), axis[:, 2]
                )
                encoded = encoded.at[:, 6:9].set(
                    jnp.stack([jnp.cos(pitch), jnp.zeros(count), -jnp.sin(pitch)], axis=-1)
                )
                encoded = encoded.at[:, 9:12].set(
                    jnp.stack(
                        [
                            jnp.sin(pitch) * jnp.cos(roll),
                            -jnp.sin(roll),
                            jnp.cos(pitch) * jnp.cos(roll),
                        ],
                        axis=-1,
                    )
                )
            encoded = encoded.at[:, :3].multiply(settings[:, 0, None])
            if not aperture_control:
                encoded = encoded.at[:, 15:18].multiply(settings[:, 3, None])
                encoded = encoded.at[:, 25:28].multiply(settings[:, 3, None])
            actor_inputs = encoded
            if "ref_mlp.0.weight" in policy:
                actor_inputs = jnp.concatenate([encoded, jax.nn.one_hot(phase, 5)], axis=-1)
            raw = actor(policy, actor_inputs)
            if lead_control:
                raw = jnp.clip(raw + settings[:, 1, None] * (raw - previous), -1.0, 1.0)
            angles = jnp.clip(
                raw[:, :2] * params["tilt_max"] * (1.0 if lead_control else settings[:, 1, None]),
                -jnp.pi / 2,
                jnp.pi / 2,
            )
            thrust = jnp.clip(
                low
                + 0.5 * (raw[:, 3] + 1) * (high - low)
                + (0.0 if aperture_control else settings[:, 2]),
                low,
                high,
            )
            if time_scaling:
                r, p = raw[:, 0] * params["tilt_max"], raw[:, 1] * params["tilt_max"]
                y = raw[:, 2] * 0.5
                cy, sy = jnp.cos(y), jnp.sin(y)
                cr, sr = jnp.cos(r), jnp.sin(r)
                cp, sp = jnp.cos(p), jnp.sin(p)
                axis = jnp.stack([cy * sp * cr + sy * sr, sy * sp * cr - cy * sr, cp * cr], axis=-1)
                base_thrust = low + 0.5 * (raw[:, 3] + 1) * (high - low)
                force = settings[:, 1, None] ** 2 * base_thrust[:, None] * axis
                force = force.at[:, 2].add((1 - settings[:, 1] ** 2) * 0.0434 * 9.81)
                magnitude = jnp.maximum(jnp.linalg.norm(force, axis=-1), 1e-6)
                axis = force / magnitude[:, None]
                roll = jnp.arcsin(jnp.clip(axis[:, 0] * sy - axis[:, 1] * cy, -1.0, 1.0))
                pitch = jnp.arctan2(axis[:, 0] * cy + axis[:, 1] * sy, axis[:, 2])
                angles = jnp.clip(jnp.stack([roll, pitch], axis=-1), -jnp.pi / 2, jnp.pi / 2)
                thrust = jnp.clip(magnitude + settings[:, 2], low, high)
            command = jnp.concatenate([angles, raw[:, 2:3] * 0.5, thrust[:, None]], axis=-1)
            data, (out, _, _, _, _) = step_env(data, command)
            obs = {k: v[:, 0] for k, v in out.items()}
            n_passed = obs["n_gates_passed"]
            finish = n_passed >= obs["gate_sequence"].shape[-1]
            times = jnp.where(~completed & finish, tick / config.env.freq, times)
            completed |= finish
            passed = jnp.maximum(passed, n_passed)
            return (data, obs, raw, thrust, completed, times, passed), None

        final, _ = jax.lax.scan(
            step, (data, obs, previous, last_thrust, completed, times, passed), jnp.arange(steps)
        )
        return tuple(x.reshape(population, trials) for x in final[-3:])

    return env, rollout


def main():
    """Search a phase calibration without using final acceptance seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=64)
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=350)
    parser.add_argument("--output", default="logs/reference-search")
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--time-scaling", action="store_true")
    parser.add_argument("--target-success", type=float, default=0.85)
    parser.add_argument("--lead-control", action="store_true")
    parser.add_argument("--aperture-control", action="store_true")
    args = parser.parse_args()
    env, rollout = build_rollout(
        args.population,
        args.trials,
        time_scaling=args.time_scaling,
        lead_control=args.lead_control,
        aperture_control=args.aperture_control,
    )
    rng = np.random.default_rng(args.seed)
    identity = np.tile([1.0, 1.0, 0.0, 1.0], (5, 1))
    if args.lead_control:
        identity[:, 1] = 0.0
    if args.aperture_control:
        identity[:, 3] = 0.0
    mean = identity.copy()
    if args.resume:
        mean = np.load(args.resume, allow_pickle=False)["gains"]
    deviation = np.tile([0.06, 0.06, 0.012, 0.07], (5, 1))
    minimum, maximum = [0.70, 0.85, -0.05, 0.75], [1.15, 1.20, 0.05, 1.40]
    if args.time_scaling:
        deviation = np.tile([0.025, 0.05, 0.004, 0.025], (5, 1))
        minimum, maximum = [0.90, 0.95, -0.02, 0.90], [1.10, 1.20, 0.02, 1.10]
    if args.lead_control:
        deviation = np.tile([0.025, 0.18, 0.004, 0.025], (5, 1))
        minimum, maximum = [0.9, 0.0, -0.02, 0.9], [1.1, 0.85, 0.02, 1.15]
    if args.aperture_control:
        deviation = np.tile([0.02, 0.025, 0.025, 0.02], (5, 1))
        minimum, maximum = [0.93, 0.95, -0.07, -0.06], [1.05, 1.08, 0.07, 0.06]
        if not args.resume:
            mean[:, 2:] = [[0.04, 0.02], [0.05, -0.03], [0.0, 0.02], [0.05, -0.02], [0.05, 0.0]]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = {"seed": args.seed, "trials": args.trials, "development_only": True, "iterations": []}
    best_score, best_gains = float("inf"), mean.copy()
    try:
        for iteration in range(args.iterations):
            gains = np.clip(
                mean + rng.normal(size=(args.population, 5, 4)) * deviation, minimum, maximum
            )
            gains[0], gains[1], gains[2] = mean, identity, best_gains
            if args.screen and iteration == 0:
                for index in range(3, args.population):
                    gains[index] = identity
                    velocity = [0.8, 0.85, 0.90, 0.95, 1.0, 1.05][(index - 3) % 6]
                    angle = [1.0, 1.05, 1.10][((index - 3) // 6) % 3]
                    gains[index, (0 if index < 21 else 1) :, 0] = velocity
                    gains[index, (0 if index < 21 else 1) :, 1] = angle
                    if args.time_scaling:
                        gains[index] = identity
                        speed = [1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 1.12][(index - 3) % 7]
                        gains[index, (0 if index < 10 else 1) :, 1] = speed
                    if args.lead_control:
                        gains[index] = identity
                        gains[index, (0 if index < 12 else 1) :, 1] = [
                            0.0,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                        ][(index - 3) % 9]
                    if args.aperture_control:
                        gains[index] = identity
                        gain = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5][(index - 3) % 7]
                        gains[index, :, 2:] = mean[:, 2:] * gain
            before = time.perf_counter()
            completed, times, passed = map(
                np.asarray,
                jax.device_get(
                    rollout(
                        jnp.asarray(gains, jnp.float32), jnp.asarray(args.seed + iteration * 1009)
                    )
                ),
            )
            success = completed.mean(axis=1)
            mean_time = np.sum(np.where(completed, times, 0.0), axis=1) / np.maximum(
                1, completed.sum(axis=1)
            )
            scores = mean_time + 20 * np.maximum(args.target_success - success, 0.0)
            scores += 0.02 * np.mean(5 - passed, axis=1)
            # Every elite is judged on the same current worlds. No best-on-a-lucky-batch selection.
            elite = np.argsort(scores)[: max(4, args.population // 6)]
            winner = int(elite[0])
            best_score, best_gains = float(scores[winner]), gains[winner].copy()
            record = {
                "iteration": iteration,
                "world_seed": args.seed + iteration * 1009,
                "seconds": time.perf_counter() - before,
                "gains": gains.tolist(),
                "success": success.tolist(),
                "times": times.tolist(),
                "scores": scores.tolist(),
                "winner": winner,
            }
            results["iterations"].append(record)
            results.update(best_gains=best_gains.tolist(), best_score=best_score)
            output.with_suffix(".json").write_text(json.dumps(results, indent=2), encoding="utf-8")
            np.savez(output.with_suffix(".npz"), gains=best_gains.astype(np.float32))
            np.savez(
                output.with_name(output.name + f"-{iteration}").with_suffix(".npz"),
                gains=best_gains.astype(np.float32),
            )
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "seconds": record["seconds"],
                        "success": float(success[winner]),
                        "score": best_score,
                        "times": times[winner].tolist(),
                        "identity_success": float(success[1]),
                        "identity_times": times[1].tolist(),
                    }
                ),
                flush=True,
            )
            mean = 0.3 * mean + 0.7 * gains[elite].mean(axis=0)
            deviation = np.maximum(
                0.3 * deviation + 0.7 * gains[elite].std(axis=0), [0.015, 0.015, 0.003, 0.02]
            )
            if args.aperture_control:
                deviation = np.minimum(deviation, [0.025, 0.03, 0.035, 0.025])
    finally:
        env.close()


if __name__ == "__main__":
    main()
