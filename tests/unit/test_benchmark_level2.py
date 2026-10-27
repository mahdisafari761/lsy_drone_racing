import unittest

from scripts.benchmark_level2 import summarize, wilson_interval


class BenchmarkStatisticsTest(unittest.TestCase):
    """Protect the benchmark's success denominator and uncertainty reporting."""

    def test_failures_do_not_count_as_fast_races(self):
        """Exclude failed-run durations from successful-race time estimates."""
        episodes = [
            {"success": True, "official_time_s": 3.0},
            {"success": False, "official_time_s": 0.4},
            {"success": True, "official_time_s": 3.4},
        ]
        result = summarize(episodes, target_time=3.2, target_success=0.8)
        self.assertAlmostEqual(result["successful_official_time_s"]["mean"], 3.2)
        self.assertAlmostEqual(result["success_rate"], 2 / 3)
        self.assertFalse(result["point_estimate_meets_target"])

    def test_no_successful_races_has_no_mean_time(self):
        """Represent absent completion times explicitly as missing."""
        result = summarize(
            [{"success": False, "official_time_s": 0.4}], target_time=3.2, target_success=0.8
        )
        self.assertIsNone(result["successful_official_time_s"]["mean"])
        self.assertFalse(result["point_estimate_meets_target"])

    def test_twenty_runs_do_not_establish_eighty_percent_with_confidence(self):
        """Retain the uncertainty associated with a small benchmark sample."""
        # 16/20 is an 80% point estimate, but its uncertainty must remain visible.
        low, high = wilson_interval(16, 20)
        self.assertAlmostEqual(low, 0.5839825677)
        self.assertAlmostEqual(high, 0.9193423374)


if __name__ == "__main__":
    unittest.main()
