"""Tests for the mandatory Ollama startup preflight."""

import unittest
from unittest.mock import Mock, patch

from src.llm.lifecycle import ensure_ollama_ready
from src.llm.ollama_client import OllamaError


class OllamaPreflightTests(unittest.TestCase):
    @patch("src.llm.lifecycle._configured_clients")
    def test_exits_when_ollama_is_unavailable(self, configured_clients) -> None:
        client = Mock()
        configured_clients.return_value = [client]
        client.available_models.side_effect = OllamaError("Ollama indisponível")

        with self.assertRaisesRegex(SystemExit, "Ollama indisponível"):
            ensure_ollama_ready()

    @patch("src.llm.lifecycle._configured_clients")
    def test_exits_with_pull_commands_for_missing_models(
        self, configured_clients
    ) -> None:
        text_client = Mock()
        vision_client = Mock()
        configured_clients.return_value = [text_client, vision_client]
        text_client.model = "qwen2.5:3b"
        vision_client.model = "qwen3-vl:2b-instruct"
        text_client.available_models.return_value = {"another-model:latest"}

        with self.assertRaises(SystemExit) as raised:
            ensure_ollama_ready()

        message = str(raised.exception)
        self.assertIn("ollama pull qwen2.5:3b", message)
        self.assertIn("ollama pull qwen3-vl:2b-instruct", message)

    @patch("src.llm.lifecycle._configured_clients")
    def test_accepts_all_configured_models(self, configured_clients) -> None:
        text_client = Mock()
        vision_client = Mock()
        configured_clients.return_value = [text_client, vision_client]
        text_client.model = "qwen2.5:3b"
        vision_client.model = "qwen3-vl:2b-instruct"
        text_client.available_models.return_value = {
            "qwen2.5:3b",
            "qwen3-vl:2b-instruct",
        }

        ensure_ollama_ready()
