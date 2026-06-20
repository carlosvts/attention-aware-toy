"""Tests for standardized profiling helpers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from src.profiling import _write_event, profile_step


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

    def test_writes_llm_readable_jsonl_event(self) -> None:
        with TemporaryDirectory() as directory:
            log_path = Path(directory) / "performance.jsonl"
            with (
                patch("src.profiling._log_directory", Path(directory)),
                patch("src.profiling._log_path", log_path),
            ):
                _write_event(
                    "step_metric",
                    "vlm",
                    {"step": "qwen_vlm_total", "metrics": {"wall_seconds": 2.7}},
                )

            event = json.loads(log_path.read_text(encoding="utf-8"))
            self.assertEqual(event["schema_version"], "1.0")
            self.assertEqual(event["component"], "vlm")
            self.assertEqual(event["step"], "qwen_vlm_total")
            self.assertEqual(event["metrics"]["wall_seconds"], 2.7)


if __name__ == "__main__":
    unittest.main()
