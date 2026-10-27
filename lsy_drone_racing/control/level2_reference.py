"""NumPy adapter of Xiaolong-Li1904's MIT-licensed Level 2 racing policy.

Source: https://github.com/Xiaolong-Li1904/lsy_drone_racing
Revision: 1cca34f2476d7791bdb7074fc9cf3dc32b4fbc27.
Adapted from rl/features.py, rl/actions.py, rl/policy_np.py, and
control/level2_rl_controller.py. Copyright (c) 2024 LSY; see the preserved
policies/level2_reference_LICENSE notice.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control import Controller
from lsy_drone_racing.envs.race_core import build_action_space

if TYPE_CHECKING:
    from ml_collections import ConfigDict


def reference_features(obs: dict, previous: np.ndarray) -> np.ndarray:
    """Encode the public observation in the reference policy's 46-feature order."""
    position = np.asarray(obs["pos"])
    rotation = Rotation.from_quat(obs["quat"]).as_matrix()
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)

    def heading(vector: np.ndarray) -> np.ndarray:
        return np.stack(
            [
                cosine * vector[..., 0] + sine * vector[..., 1],
                -sine * vector[..., 0] + cosine * vector[..., 1],
                vector[..., 2],
            ],
            axis=-1,
        )

    passed = int(obs["n_gates_passed"])
    passes = len(obs["gate_sequence"])

    def gate(advance: int) -> np.ndarray:
        slot = min(passed + advance, passes - 1)
        index = int(obs["gate_sequence"][slot])
        sign = int(obs["gate_sequence_direction"][slot])
        matrix = Rotation.from_quat(obs["gates_quat"][index]).as_matrix()
        return np.r_[
            heading(obs["gates_pos"][index] - position),
            heading(matrix[:, 0] * sign),
            heading(matrix[:, 1] * sign),
            float(obs["gates_visited"][index]),
        ]

    second = passed + 1 < passes
    poles = np.asarray(obs["obstacles_pos"])[:, :2] - position[None, :2]
    order = np.argsort(np.sum(poles * poles, axis=-1))[:2]
    near = np.c_[poles[order], np.zeros(len(order))]
    pillars = np.c_[heading(near)[:, :2], np.asarray(obs["obstacles_visited"])[order]]
    # Level 2 contains four poles; padding keeps the feature shape valid on simpler tracks.
    if len(pillars) < 2:
        pillars = np.concatenate([pillars, np.zeros((2 - len(pillars), 3))])
    return np.r_[
        heading(np.asarray(obs["vel"])),
        obs["ang_vel"],
        heading(rotation[:, 0]),
        heading(rotation[:, 2]),
        sine,
        cosine,
        position[2],
        gate(0),
        gate(1) * second,
        second,
        pillars.ravel(),
        previous,
    ].astype(np.float32)


class Level2Reference(Controller):
    """Apply a public racing warm start using only observable states and objects."""

    POLICY_FILE = "level2_reference.npz"

    def __init__(self, obs: dict, info: dict, config: ConfigDict):
        """Load normalized policy arrays and the public action bounds."""
        super().__init__(obs, info, config)
        if config.env.control_mode != "attitude":
            raise ValueError("Level2Reference requires the attitude interface")
        with np.load(
            Path(__file__).parent / "policies" / self.POLICY_FILE, allow_pickle=False
        ) as data:
            self.mean = data["obs_mean"].astype(np.float64)
            self.std = data["obs_std"].astype(np.float64)
            self.layers = [
                (data[f"W{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
                for i in range(int(data["n_layers"]))
            ]
            self.tilt_max = float(data["tilt_max"]) if "tilt_max" in data else 1.0
        space = build_action_space("attitude", config.sim.drone, config.sim.dynamics)
        self.thrust_low, self.thrust_high = float(space.low[3]), float(space.high[3])
        self.previous = np.zeros(4)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return bounded roll, pitch, yaw and collective thrust commands."""
        value = reference_features(obs, self.previous)
        value = np.clip((value - self.mean) / self.std, -10.0, 10.0)
        for index, (weight, bias) in enumerate(self.layers):
            value = value @ weight + bias
            if index < len(self.layers) - 1:
                value = np.where(value > 0, value, np.expm1(np.minimum(value, 0.0)))
        raw = np.clip(value, -1.0, 1.0)
        self.previous = raw.copy()
        thrust = self.thrust_low + 0.5 * (raw[3] + 1) * (self.thrust_high - self.thrust_low)
        return np.array(
            [raw[0] * self.tilt_max, raw[1] * self.tilt_max, raw[2] * 0.5, thrust], dtype=np.float32
        )

    def episode_callback(self):
        """Reset action history when a race ends."""
        self.previous = np.zeros(4)

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Let the unchanged racing environment decide episode termination."""
        return False
