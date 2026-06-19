"""Tests for the Ollama HTTP client lifecycle."""

import unittest
from unittest.mock import patch

from src.llm.ollama_client import OllamaClient


class OllamaClientLifecycleTests(unittest.TestCase):
    @patch("src.llm.ollama_client.requests.post")
    def test_unload_requests_immediate_model_eviction(self, post) -> None:
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        client.unload()

        post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={"model": "qwen-test", "keep_alive": 0},
            timeout=(2.0, 10.0),
        )
        post.return_value.raise_for_status.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
