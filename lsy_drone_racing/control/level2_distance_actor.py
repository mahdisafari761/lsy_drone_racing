"""Prototype a public-distance mixture of original and adapted racing actors.

The default archives point to local experiment artifacts for ordinary-controller
validation. This prototype is not a portable release controller. It retains the
original race physics, observations and shared Lead action history.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from lsy_drone_racing.control.level2_phase_actor import Level2PhaseActor
from lsy_drone_racing.control.level2_reference import reference_features

_ROOT = Path(__file__).resolve().parents[2]


class Level2DistanceActor(Level2PhaseActor):
    """Interpolate two cached actors using only the current public gate distance."""

    POLICY_FILE = str(_ROOT / "logs/lead_v2/fast-early-bundle.npz")
    GAINS_FILE = str(_ROOT / "logs/lead_v3/fast-early-rotating/candidate-004-15.npz")
    TRAINED_PHASES = (0, 1)
    ADAPTED_PHASES = (0, 1)
    ADAPTED_FAR = True
    MIX_NEAR = 0.6
    MIX_FAR = 1.6

    def __init__(self, obs: dict, info: dict, config: dict):
        """Validate selected sections and increasing public-distance thresholds."""
        super().__init__(obs, info, config)
        phases = tuple(self.ADAPTED_PHASES)
        if any(not isinstance(phase, (int, np.integer)) or not 0 <= phase < 5 for phase in phases):
            raise ValueError("ADAPTED_PHASES must contain section indices from 0 to 4")
        self.adapted_phases = frozenset(phases)
        self.mix_near, self.mix_far = float(self.MIX_NEAR), float(self.MIX_FAR)
        if not (
            np.isfinite(self.mix_near)
            and np.isfinite(self.mix_far)
            and 0 <= self.mix_near < self.mix_far
        ):
            raise ValueError("Mixture distances must satisfy finite 0 <= near < far")
        self.cached_actors = (self.base_layers, self.trained_layers)

    def compute_control(self, obs: dict, info: dict | None = None) -> np.ndarray:
        """Mix individually clipped actors before the single Lead/history update."""
        phase = min(int(obs["n_gates_passed"]), 4)
        encoded = reference_features(obs, self.previous)
        weight = 0.0
        if phase in self.adapted_phases:
            distance = float(np.linalg.norm(encoded[15:18]))
            fraction = float(
                np.clip((distance - self.mix_near) / (self.mix_far - self.mix_near), 0, 1)
            )
            smooth = fraction * fraction * (3 - 2 * fraction)
            weight = smooth if self.ADAPTED_FAR else 1 - smooth
        velocity_scale, lead, thrust_bias, distance_scale = self.gains[phase]
        encoded[:3] *= velocity_scale
        encoded[15:18] *= distance_scale
        encoded[25:28] *= distance_scale
        normalized = np.clip((encoded - self.mean) / self.std, -10, 10)
        raw = np.zeros(4)
        for probability, layers in zip((1 - weight, weight), self.cached_actors, strict=True):
            if probability == 0:
                continue
            value = normalized
            for index, (matrix, intercept) in enumerate(layers):
                value = value @ matrix + intercept
                if index < len(layers) - 1:
                    value = np.where(value > 0, value, np.expm1(np.minimum(value, 0)))
            raw += probability * np.clip(value, -1, 1)
        raw = np.clip(raw + lead * (raw - self.previous), -1, 1)
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
