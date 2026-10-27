"""Verify that vectorized tuning evaluates the deployed reference observation/actor."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401 - configure SciPy array API before importing SciPy
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from lsy_drone_racing.control._reference_policy_jax import actor, load_policy
from lsy_drone_racing.control._reference_policy_jax import reference_features as batched_features
from lsy_drone_racing.control.level2_lead import Level2Lead
from lsy_drone_racing.control.level2_reference import Level2Reference, reference_features
from lsy_drone_racing.utils import load_config

ROOT = Path(__file__).resolve().parents[2]


def observation(phase: int) -> dict:
    """Build public observations with nonzero roll/yaw and a repeated reverse gate."""
    return {
        "pos": np.array([0.1, -0.3, 0.8]),
        "vel": np.array([1.2, -0.6, 0.4]),
        "ang_vel": np.array([0.5, -0.2, 0.1]),
        "quat": Rotation.from_euler("xyz", [0.3, -0.4, 0.8]).as_quat(),
        "gates_pos": np.array(
            [[0.5, 0.25, 0.7], [1.05, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]]
        ),
        "gates_quat": Rotation.from_euler(
            "xyz", [[0.02, 0.03, -0.78], [0.01, -0.02, 2.35], [0.0, 0.02, 3.14], [0.01, 0.0, 0.0]]
        ).as_quat(),
        "gate_sequence": np.array([0, 1, 2, 3, 1]),
        "gate_sequence_direction": np.array([1, -1, 1, 1, -1]),
        "n_gates_passed": np.asarray(phase),
        "gates_visited": np.array([True, phase > 0, phase > 1, phase > 2]),
        "obstacles_pos": np.array(
            [[0.0, 0.75, 1.55], [1.0, 0.25, 1.55], [-1.5, -0.25, 1.55], [-0.5, -0.75, 1.55]]
        ),
        "obstacles_visited": np.array([True, False, True, False]),
    }


class TestReferencePolicy(unittest.TestCase):
    """Keep public-feature and normalized-action semantics equal across backends."""

    def test_batched_encoder_and_actor(self):
        """Compare signed/repeated gate features and actual exported actor outputs."""
        config = load_config(ROOT / "config/level2.toml")
        controller = Level2Reference(observation(0), {}, config)
        params = load_policy(ROOT / "lsy_drone_racing/control/policies/level2_reference.npz")
        for phase in range(5):
            obs = observation(phase)
            previous = controller.previous.copy()
            encoded = batched_features(
                {k: jnp.asarray(v)[None] for k, v in obs.items()}, jnp.asarray(previous)[None]
            )
            np.testing.assert_allclose(encoded[0], reference_features(obs, previous), atol=1e-6)
            raw = np.asarray(actor(params, encoded))[0]
            action = controller.compute_control(obs)
            expected = np.r_[
                raw[:2] * controller.tilt_max,
                raw[2] * 0.5,
                controller.thrust_low
                + 0.5 * (raw[3] + 1) * (controller.thrust_high - controller.thrust_low),
            ]
            np.testing.assert_allclose(action, expected, atol=2e-5)

    def test_identity_calibration_preserves_action(self):
        """Keep the uncalibrated actor identical when the calibration is disabled."""
        config = load_config(ROOT / "config/level2.toml")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.npz"
            np.savez(path, gains=np.tile([1.0, 0.0, 0.0, 1.0], (5, 1)))

            class IdentityScaled(Level2Lead):
                GAINS_FILE = str(path)

            original = Level2Reference(observation(0), {}, config)
            scaled = IdentityScaled(observation(0), {}, config)
            for phase in range(5):
                obs = observation(phase)
                np.testing.assert_allclose(
                    original.compute_control(obs), scaled.compute_control(obs), atol=3e-6
                )

    def test_calibrated_search_matches_deployment(self):
        """Use the same action history, phase gains and limits in search and deployment."""
        config = load_config(ROOT / "config/level2.toml")
        controller = Level2Lead(observation(0), {}, config)
        params = load_policy(ROOT / "lsy_drone_racing/control/policies/level2_reference.npz")
        for phase in range(5):
            obs = observation(phase)
            previous = controller.previous.copy()
            velocity, lead, bias, distance = controller.gains[phase]
            encoded = batched_features(
                {k: jnp.asarray(v)[None] for k, v in obs.items()}, jnp.asarray(previous)[None]
            )
            encoded = encoded.at[:, :3].multiply(velocity)
            encoded = encoded.at[:, 15:18].multiply(distance)
            encoded = encoded.at[:, 25:28].multiply(distance)
            raw = np.asarray(actor(params, encoded))[0]
            raw = np.clip(raw + lead * (raw - previous), -1.0, 1.0)
            expected = np.r_[
                raw[:2] * controller.tilt_max,
                raw[2] * 0.5,
                np.clip(
                    controller.thrust_low
                    + 0.5 * (raw[3] + 1) * (controller.thrust_high - controller.thrust_low)
                    + bias,
                    controller.thrust_low,
                    controller.thrust_high,
                ),
            ]
            action = controller.compute_control(obs)
            np.testing.assert_allclose(action, expected, atol=2e-5)
            self.assertTrue(np.all(np.isfinite(action)))


if __name__ == "__main__":
    unittest.main()
