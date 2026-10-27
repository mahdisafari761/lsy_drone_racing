"""Bias the reference target inside gate openings to correct systematic edge approaches."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control.level2_lead import Level2Lead


class Level2GuardedLead(Level2Lead):
    """Keep gate target offsets small and local while preserving learned race dynamics."""

    STRENGTH = 0.0
    BLIND_SPEED_GAIN = 1.1
    LATERAL_OFFSETS = (0.04, -0.025, 0.035, 0.035, -0.02)
    VERTICAL_OFFSETS = (0.015, 0.0, 0.0, 0.0, 0.0)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Shift only the neural target points, retaining the original observed frame normals."""
        adjusted = dict(obs)
        first = int(obs["gate_sequence"][0])
        if int(obs["n_gates_passed"]) == 0 and not obs["gates_visited"][first]:
            delta = np.asarray(obs["gates_pos"])[first, :2] - np.asarray(obs["pos"])[:2]
            distance = np.linalg.norm(delta)
            strength = np.clip((1.5 - distance) / 0.7, 0.0, 1.0)
            scale = 1.0 + strength * (self.BLIND_SPEED_GAIN - 1.0)
            adjusted["vel"] = np.asarray(obs["vel"]) * [scale, scale, 1.0]
        targets = np.asarray(obs["gates_pos"]).copy()
        phase = int(obs["n_gates_passed"])
        for index in range(phase, min(phase + 2, len(obs["gate_sequence"]))):
            gate = int(obs["gate_sequence"][index])
            rotation = Rotation.from_quat(obs["gates_quat"][gate]).as_matrix()
            offset = rotation[:, 1] * self.LATERAL_OFFSETS[index]
            offset += rotation[:, 2] * self.VERTICAL_OFFSETS[index]
            targets[gate] += self.STRENGTH * offset
        adjusted["gates_pos"] = targets
        return super().compute_control(adjusted, info)
