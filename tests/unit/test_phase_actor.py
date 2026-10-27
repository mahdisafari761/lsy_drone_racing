"""Check phase switching against independent actors with the same applied history."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401 - configure SciPy array API before importing SciPy
import numpy as np

from lsy_drone_racing.control.level2_lead import Level2Lead
from lsy_drone_racing.control.level2_phase_actor import Level2PhaseActor
from lsy_drone_racing.utils import load_config
from tests.unit.test_reference_policy import observation

ROOT = Path(__file__).resolve().parents[2]


class TestPhaseActor(unittest.TestCase):
    """Preserve exact actor outputs, public gate semantics and shared action history."""

    def test_phase_selection_matches_independent_actors(self):
        """Compare every section through signed/repeated gates and multiple switches."""
        config = load_config(ROOT / "config/level2.toml")
        policy_dir = ROOT / "lsy_drone_racing/control/policies"
        with np.load(policy_dir / "level2_reference.npz", allow_pickle=False) as archive:
            base = {key: archive[key] for key in archive.files}
        checkpoint = ROOT / "logs/lead_v2/actor-ppo-00409600.npz"
        if checkpoint.is_file():
            # Development also checks the actual fine-tuned weights. A fresh
            # checkout uses a deterministic perturbation to exercise both paths.
            with np.load(checkpoint, allow_pickle=False) as archive:
                adapted = {key: archive[key] for key in archive.files}
        else:
            adapted = {key: value.copy() for key, value in base.items()}
            adapted["b3"] = adapted["b3"] + np.asarray([0.03, -0.04, 0.02, -0.05])
        bundle = dict(base)
        bundle.update(
            {
                f"trained_{name}{index}": adapted[f"{name}{index}"]
                for index in range(int(base["n_layers"]))
                for name in ("W", "b")
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            base_path, adapted_path, bundle_path = [
                Path(directory) / filename for filename in ("base.npz", "adapted.npz", "bundle.npz")
            ]
            np.savez(base_path, **base)
            np.savez(adapted_path, **adapted)
            np.savez(bundle_path, **bundle)

            class BaseActor(Level2Lead):
                POLICY_FILE = str(base_path)

            class AdaptedActor(Level2Lead):
                POLICY_FILE = str(adapted_path)

            for phases in ((), (0, 1, 2, 3, 4), (1, 2, 3)):
                with self.subTest(trained_phases=phases):

                    class SelectedActor(Level2PhaseActor):
                        POLICY_FILE = str(bundle_path)
                        TRAINED_PHASES = phases

                    selected = SelectedActor(observation(0), {}, config)
                    base_actor = BaseActor(observation(0), {}, config)
                    adapted_actor = AdaptedActor(observation(0), {}, config)
                    previous = np.zeros(4)
                    for step, phase in enumerate((0, 0, 1, 1, 2, 3, 4, 4, 1, 0)):
                        public = observation(phase)
                        public["vel"] = public["vel"] + 0.03 * step
                        expected_actor = adapted_actor if phase in phases else base_actor
                        expected_actor.previous = previous.copy()
                        expected = expected_actor.compute_control(public)
                        actual = selected.compute_control(public)
                        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=0.0)
                        np.testing.assert_allclose(
                            selected.previous, expected_actor.previous, atol=3e-7, rtol=0.0
                        )
                        previous = expected_actor.previous.copy()
                        self.assertTrue(np.all(np.isfinite(actual)))
                        self.assertTrue(np.all(np.abs(actual[:2]) <= selected.tilt_max))
                        self.assertLessEqual(abs(actual[2]), 0.5)
                        self.assertGreaterEqual(actual[3], selected.thrust_low - 1e-7)
                        self.assertLessEqual(actual[3], selected.thrust_high + 1e-7)
                    selected.episode_callback()
                    np.testing.assert_array_equal(selected.previous, np.zeros(4))


if __name__ == "__main__":
    unittest.main()
