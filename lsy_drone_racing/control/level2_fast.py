"""Sequence-aware NumPy adaptation of Yuming Li's public PPO racing policy.

Policy and observation definition: Yuming-Lee24/lsy_drone_racing, 25ws-main,
my_rl_controller.py and lv2_fast_policy.ckpt. MIT license, copyright 2024 LSY.
See policies/LICENSE. The adapter supports the current signed gate sequence,
including repeated physical gates, and uses public observations exclusively.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from ml_collections import ConfigDict

_CORNERS = np.array([[0, -0.2, 0.2], [0, 0.2, 0.2], [0, 0.2, -0.2], [0, -0.2, -0.2]])


class Level2Fast(Controller):
    """Apply a learned racing policy without a PyTorch runtime dependency."""

    POLICY_FILE = "level2_fast.npz"

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Load the deterministic actor and initialize its two-step history."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("Level2Fast requires the attitude interface")
        with np.load(
            Path(__file__).parent / "policies" / self.POLICY_FILE, allow_pickle=False
        ) as data:
            self.layers = [
                (data[f"actor_mean.{i}.weight"], data[f"actor_mean.{i}.bias"]) for i in [0, 2, 4]
            ]
        self.residual = None
        self.phase_residual = None
        self.mlp = None
        self.mlp_phase = None
        self.initial_layers = None
        self.initial_residual = None
        with np.load(
            Path(__file__).parent / "policies" / self.POLICY_FILE, allow_pickle=False
        ) as data:
            if "residual.weight" in data:
                self.residual = (data["residual.weight"], data["residual.bias"])
            if "phase_residual.weight" in data:
                self.phase_residual = (data["phase_residual.weight"], data["phase_residual.bias"])
            if "mlp.0.weight" in data:
                self.mlp = [(data[f"mlp.{i}.weight"], data[f"mlp.{i}.bias"]) for i in [0, 1]]
            if "initial_actor_mean.0.weight" in data:
                self.initial_layers = [
                    (data[f"initial_actor_mean.{i}.weight"], data[f"initial_actor_mean.{i}.bias"])
                    for i in [0, 2, 4]
                ]
                self.initial_residual = (
                    data["initial_residual.weight"],
                    data["initial_residual.bias"],
                )
            if "mlp_phase" in data:
                self.mlp_phase = int(data["mlp_phase"])
        self.previous_action = np.zeros(4, dtype=np.float32)
        self.pending_action = np.zeros(4, dtype=np.float32)
        self.history = np.tile(self.basic_state(obs), (2, 1))
        self.pending_state = None
        self.action_low = np.array([-np.pi / 2] * 3 + [0.2], dtype=np.float32)
        self.action_high = np.array([np.pi / 2] * 3 + [0.8], dtype=np.float32)
        self.tick = 0
        self.phase = "race"
        self.target = np.asarray(obs["pos"]).copy()

    @staticmethod
    def basic_state(obs: dict) -> np.ndarray:
        """Encode the historical world-frame state in the training order."""
        rotation = Rotation.from_quat(obs["quat"]).as_matrix()
        return np.r_[obs["pos"][2], rotation.ravel(), obs["vel"], obs["ang_vel"]].astype(np.float32)

    def features(self, obs: dict) -> np.ndarray:
        """Encode body-relative gate corners, poles and delayed state history."""
        if self.pending_state is not None:
            self.history[:-1] = self.history[1:]
            self.history[-1] = self.pending_state
        pos = np.asarray(obs["pos"])
        rotation = Rotation.from_quat(obs["quat"]).as_matrix()
        passed = int(obs["n_gates_passed"])
        corners = []
        for offset in [0, 1]:
            index = min(passed + offset, len(obs["gate_sequence"]) - 1)
            gid = int(obs["gate_sequence"][index])
            gate_rotation = Rotation.from_quat(obs["gates_quat"][gid]).as_matrix()
            local = _CORNERS.copy()
            local[:, 1] *= int(obs["gate_sequence_direction"][index])
            world = local @ gate_rotation.T + obs["gates_pos"][gid]
            corners.append(((world - pos) @ rotation).ravel())
        poles = np.asarray(obs["obstacles_pos"]).copy()
        obstacle_features = np.full(12, 10.0, dtype=np.float32)
        if len(poles):
            poles[:, 2] = np.minimum(poles[:, 2], pos[2])
            body = (poles - pos) @ rotation
            order = np.argsort(np.linalg.norm(body, axis=1))[:4]
            obstacle_features[: 3 * len(order)] = body[order].ravel()
        result = np.r_[
            pos[2],
            np.asarray(obs["vel"]) @ rotation,
            obs["ang_vel"],
            rotation.ravel(),
            corners[0],
            corners[1],
            self.previous_action,
            obstacle_features,
            self.history.ravel(),
        ].astype(np.float32)
        self.pending_state = self.basic_state(obs)
        return result

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Run the actor and return bounded roll, pitch, yaw and thrust commands."""
        x = self.features(obs)
        raw = x.copy()
        hidden = None
        initial = self.initial_layers is not None and int(obs["n_gates_passed"]) == 0
        layers = self.initial_layers if initial else self.layers
        for index, (weight, bias) in enumerate(layers):
            x = np.tanh(weight @ x + bias)
            if index == 1:
                hidden = x.copy()
        residual = self.initial_residual if initial else self.residual
        if residual is not None:
            weight, bias = residual
            x = x + 0.5 * np.tanh(weight @ np.r_[raw, hidden] + bias)
        if self.phase_residual is not None and not initial:
            weight, bias = self.phase_residual
            phase = min(int(obs["n_gates_passed"]), len(weight) - 1)
            x = x + 0.5 * np.tanh(weight[phase] @ np.r_[raw, hidden] + bias[phase])
        if self.mlp is not None and (
            self.mlp_phase is None or int(obs["n_gates_passed"]) == self.mlp_phase
        ):
            phase = np.eye(5, dtype=np.float32)[min(int(obs["n_gates_passed"]), 4)]
            encoded = np.r_[raw, hidden, phase, np.asarray(obs["gates_visited"], dtype=np.float32)]
            for weight, bias in self.mlp:
                if weight.ndim == 3:
                    index = min(int(obs["n_gates_passed"]), len(weight) - 1)
                    weight, bias = weight[index], bias[index]
                encoded = np.tanh(weight @ encoded + bias)
            x = x + 0.5 * encoded
        self.previous_action = self.pending_action.copy()
        self.pending_action = x.copy()
        action = self.action_low + (np.clip(x, -1, 1) + 1) * 0.5 * (
            self.action_high - self.action_low
        )
        self.tick += 1
        index = min(int(obs["n_gates_passed"]), len(obs["gate_sequence"]) - 1)
        self.target = np.asarray(obs["gates_pos"])[int(obs["gate_sequence"][index])].copy()
        return action.astype(np.float32)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Finish only after all signed gate entries have been passed."""
        return int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])
