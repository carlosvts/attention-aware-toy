"""Tests for the Ollama HTTP client lifecycle."""

import unittest
from unittest.mock import patch

from src.llm.ollama_client import OllamaClient


class OllamaClientLifecycleTests(unittest.TestCase):
    @patch("src.llm.ollama_client.record_elapsed")
    @patch("src.llm.ollama_client.requests.post")
    def test_chat_streams_and_records_first_token_and_total(
        self, post, record_elapsed
    ) -> None:
        response = post.return_value
        response.iter_lines.return_value = [
            '{"message":{"content":"Olá"},"done":false}',
            '{"message":{"content":"!"},"done":true}',
        ]
        client = OllamaClient("http://localhost:11434", "qwen-test", 30.0)

        result = client.chat("system", "user", profiling_name="qwen-test")

        self.assertEqual(result, "Olá!")
        self.assertTrue(post.call_args.kwargs["stream"])
        self.assertTrue(post.call_args.kwargs["json"]["stream"])
        self.assertEqual(
            [item.args[0] for item in record_elapsed.call_args_list],
            ["qwen-test_first_token", "qwen-test_total"],
        )
        response.close.assert_called_once_with()

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
