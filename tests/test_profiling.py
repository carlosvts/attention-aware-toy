"""Tests for standardized profiling helpers."""

import unittest
from unittest.mock import patch

from src.profiling import profile_step


class ProfilingTests(unittest.TestCase):
    @patch("src.profiling.record_step")
    @patch("src.profiling.time.process_time", side_effect=[4.0, 4.25])
    @patch("src.profiling.time.perf_counter", side_effect=[10.0, 10.5])
    def test_profile_step_records_wall_and_cpu_time(
        self, perf_counter, process_time, record_step
    ) -> None:
        @profile_step("example")
        def operation(value: int) -> int:
            return value * 2

        self.assertEqual(operation(3), 6)
        record_step.assert_called_once_with("example", 0.5, 0.25)


if __name__ == "__main__":
    unittest.main()
