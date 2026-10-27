"""Keep release acceptance stricter than a rounded eighty-percent score."""

import unittest

from scripts.validate_lead_release import acceptance, release_manifest


def races(successes: int, total: int, completion_time: float) -> dict:
    """Create a small result fixture including official gate split semantics."""
    return {
        "episodes": [
            {
                "success": index < successes,
                "official_time_s": completion_time,
                "gates_passed": 5 if index < successes else 2,
                "gate_pass_times_s": [completion_time * part / 5 for part in range(1, 6)],
            }
            for index in range(total)
        ]
    }


class ReleaseAcceptanceTest(unittest.TestCase):
    """Test the decision boundaries and distinguish sample results from confidence."""

    def test_exact_eighty_percent_fails_strict_success_target(self):
        """A point estimate of exactly eighty percent does not exceed eighty percent."""
        result = acceptance(races(16, 20, 3.29), 3.30, 0.8)
        self.assertFalse(result["strict_empirical_target_met"])

    def test_time_and_success_must_both_pass(self):
        """Reject a too-slow controller even when its success rate is sufficient."""
        self.assertFalse(acceptance(races(17, 20, 3.31), 3.30, 0.8)["strict_empirical_target_met"])
        passed = acceptance(races(17, 20, 3.30), 3.30, 0.8)
        self.assertTrue(passed["strict_empirical_target_met"])
        self.assertFalse(passed["success_lower_confidence_bound_exceeds_target"])

    def test_manifest_includes_inherited_runtime_and_arrays(self):
        """Freeze the inference adapter and base class, not only the small entry-point file."""
        manifest = release_manifest("level2_lead.py")
        source_names = manifest["controller_sources"]
        self.assertTrue(any(name.endswith("level2_reference.py") for name in source_names))
        self.assertTrue(any(name.endswith("controller.py") for name in source_names))
        self.assertEqual(set(manifest["model_artifacts"]), {"POLICY_FILE", "GAINS_FILE"})

    def test_two_equal_quantized_times_do_not_prove_mean_time(self):
        """A two-race smoke test must not emit a zero-width confidence certificate."""
        result = acceptance(races(2, 2, 3.26), 3.30, 0.8)
        self.assertIsNone(result["successful_mean_time_95_one_sided_upper_s"])
        self.assertFalse(result["time_upper_confidence_bound_meets_target"])


if __name__ == "__main__":
    unittest.main()
