"""Check distance-mixture commands against independently evaluated Lead actors."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import crazyflow  # noqa: F401 - configure SciPy array API before importing SciPy
import numpy as np

from lsy_drone_racing.control.level2_distance_actor import Level2DistanceActor
from lsy_drone_racing.control.level2_lead import Level2Lead
from lsy_drone_racing.utils import load_config
from tests.unit.test_reference_policy import observation

ROOT = Path(__file__).resolve().parents[2]


class TestLevel2DistanceActor(unittest.TestCase):
    """Preserve clipping order, public scheduling and the common action history."""

    def test_matches_independent_actors_and_one_lead_update(self):
        """Compare near, middle, far and unselected phases in both interpolation directions."""
        config = load_config(ROOT / "config/level2.toml")
        with np.load(
            ROOT / "lsy_drone_racing/control/policies/level2_best.npz", allow_pickle=False
        ) as archive:
            bundle = {key: archive[key] for key in archive.files}
        with tempfile.TemporaryDirectory() as directory:
            bundle_path = Path(directory) / "bundle.npz"
            np.savez(bundle_path, **bundle)
            actor_paths = []
            for prefix in ("", "trained_"):
                arrays = dict(bundle)
                for index in range(int(bundle["n_layers"])):
                    for name in ("W", "b"):
                        arrays[f"{name}{index}"] = bundle[f"{prefix}{name}{index}"]
                path = Path(directory) / f"actor-{len(actor_paths)}.npz"
                np.savez(path, **arrays)
                actor_paths.append(path)
            actors = []
            for path in actor_paths:

                class Independent(Level2Lead):
                    POLICY_FILE = str(path)
                    GAINS_FILE = "level2_best_gains.npz"

                actor = Independent(observation(0), {}, config)
                actor.gains[:, 1:3] = 0
                actors.append(actor)
            for adapted_far in (False, True):

                class Mixture(Level2DistanceActor):
                    POLICY_FILE = str(bundle_path)
                    GAINS_FILE = "level2_best_gains.npz"
                    ADAPTED_FAR = adapted_far

                mixture = Mixture(observation(0), {}, config)
                previous = np.zeros(4)
                for phase, distance in ((0, 0.2), (0, 1.1), (1, 2.0), (1, 0.6), (2, 1.1), (4, 2.0)):
                    public = observation(phase)
                    gate = int(public["gate_sequence"][phase])
                    public["pos"] = public["gates_pos"][gate] - [distance, 0, 0]
                    fraction = np.clip((distance - 0.6) / (1.6 - 0.6), 0, 1)
                    smooth = fraction**2 * (3 - 2 * fraction)
                    weight = (smooth if adapted_far else 1 - smooth) if phase in (0, 1) else 0
                    values = []
                    for actor in actors:
                        actor.previous = previous.copy()
                        command = actor.compute_control(public)
                        values.append(
                            np.r_[
                                command[:2] / actor.tilt_max,
                                command[2] / 0.5,
                                2
                                * (command[3] - actor.thrust_low)
                                / (actor.thrust_high - actor.thrust_low)
                                - 1,
                            ]
                        )
                    raw = (1 - weight) * values[0] + weight * values[1]
                    raw = np.clip(raw + mixture.gains[phase, 1] * (raw - previous), -1, 1)
                    thrust = mixture.thrust_low + 0.5 * (raw[3] + 1) * (
                        mixture.thrust_high - mixture.thrust_low
                    )
                    expected = np.r_[
                        raw[:2] * mixture.tilt_max,
                        raw[2] * 0.5,
                        np.clip(
                            thrust + mixture.gains[phase, 2],
                            mixture.thrust_low,
                            mixture.thrust_high,
                        ),
                    ]
                    actual = mixture.compute_control(public)
                    np.testing.assert_allclose(actual, expected, atol=7e-6, rtol=0)
                    np.testing.assert_allclose(mixture.previous, raw, atol=7e-6, rtol=0)
                    previous = mixture.previous.copy()
                mixture.episode_callback()
                np.testing.assert_array_equal(mixture.previous, np.zeros(4))

    def test_unselected_phases_ignore_adapted_actor(self):
        """Use the ordinary actor exactly when the phase mask is empty."""
        config = load_config(ROOT / "config/level2.toml")

        class Mixture(Level2DistanceActor):
            POLICY_FILE = "level2_best.npz"
            GAINS_FILE = "level2_best_gains.npz"
            ADAPTED_PHASES = ()

        class Baseline(Level2Lead):
            POLICY_FILE = "level2_best.npz"
            GAINS_FILE = "level2_best_gains.npz"

        mixture, baseline = (
            Mixture(observation(0), {}, config),
            Baseline(observation(0), {}, config),
        )
        for phase in range(5):
            public = observation(phase)
            np.testing.assert_array_equal(
                mixture.compute_control(public), baseline.compute_control(public)
            )
            np.testing.assert_array_equal(mixture.previous, baseline.previous)


if __name__ == "__main__":
    unittest.main()
