"""Phase-calibrated racing policy with a bounded action lead filter.

The actor is adapted from Xiaolong-Li1904's MIT-licensed public reference;
see level2_reference.py and policies/level2_reference_LICENSE for attribution.
Only public observations, nominal action bounds and frozen development gains
are used during a race. No race-simulator data or random seeds are accessed.
"""

from pathlib import Path

import numpy as np

from lsy_drone_racing.control.level2_reference import Level2Reference, reference_features


class Level2Lead(Level2Reference):
    """Calibrate observable features by section and compensate a little servo lag."""

    GAINS_FILE = "level2_lead_gains.npz"

    def __init__(self, obs: dict, info: dict, config: dict):
        """Load the frozen calibration independently of the live race state."""
        super().__init__(obs, info, config)
        with np.load(
            Path(__file__).parent / "policies" / self.GAINS_FILE, allow_pickle=False
        ) as archive:
            self.gains = np.asarray(archive["gains"], dtype=np.float64)
        if self.gains.shape != (5, 4) or not np.all(np.isfinite(self.gains)):
            raise ValueError("Expected finite phase calibration with shape (5, 4)")

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Return roll, pitch, yaw and collective thrust within the original bounds."""
        phase = min(int(obs["n_gates_passed"]), 4)
        velocity_scale, lead, thrust_bias, distance_scale = self.gains[phase]
        encoded = reference_features(obs, self.previous)
        encoded[:3] *= velocity_scale
        encoded[15:18] *= distance_scale
        encoded[25:28] *= distance_scale
        value = np.clip((encoded - self.mean) / self.std, -10.0, 10.0)
        for index, (weight, intercept) in enumerate(self.layers):
            value = value @ weight + intercept
            if index < len(self.layers) - 1:
                value = np.where(value > 0, value, np.expm1(np.minimum(value, 0.0)))
        raw = np.clip(value, -1.0, 1.0)
        raw = np.clip(raw + lead * (raw - self.previous), -1.0, 1.0)
        self.previous = raw.copy()
        thrust = self.thrust_low + 0.5 * (raw[3] + 1) * (self.thrust_high - self.thrust_low)
        return np.asarray(
            [
                raw[0] * self.tilt_max,
                raw[1] * self.tilt_max,
                raw[2] * 0.5,
                np.clip(thrust + thrust_bias, self.thrust_low, self.thrust_high),
            ],
            dtype=np.float32,
        )

    def step_callback(
        self,
        action: np.ndarray,
        obs: dict,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Finish immediately after the complete configured gate sequence."""
        return int(obs["n_gates_passed"]) >= len(obs["gate_sequence"])
