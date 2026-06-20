"""Tests for separate VLM instrumentation."""

import unittest
from unittest.mock import patch

import numpy as np

from src.vision.scene_describer import describe_scene


class SceneDescriberProfilingTests(unittest.TestCase):
    @patch("src.profiling.record_step")
    @patch("src.vision.scene_describer._encode_frame", return_value="image")
    @patch("src.vision.scene_describer.OllamaClient.from_environment")
    def test_labels_visual_model_as_vlm(
        self, client_factory, encode_frame, record_step
    ) -> None:
        client_factory.return_value.chat.return_value = "Uma pessoa."

        result = describe_scene(np.zeros((2, 2, 3), dtype=np.uint8))

        self.assertEqual(result, "Uma pessoa.")
        self.assertEqual(
            client_factory.return_value.chat.call_args.kwargs["profiling_name"],
            "qwen_vlm",
        )


if __name__ == "__main__":
    unittest.main()
