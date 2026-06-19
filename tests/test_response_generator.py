"""Tests for the contextual local-LLM prompt."""

import unittest
from unittest.mock import patch

from src.attention import AttentionState
from src.llm.response_generator import generate_response


class ResponsePromptTests(unittest.TestCase):
    @patch("src.llm.response_generator.OllamaClient.from_environment")
    def test_prompt_contains_attention_duration_and_scene(self, client_factory) -> None:
        client_factory.return_value.chat.return_value = "Olá!"

        result = generate_response(
            "Uma pessoa está diante da câmera.",
            AttentionState.ATTENDING,
            1.24,
        )

        self.assertEqual(result, "Olá!")
        system_prompt, user_prompt = client_factory.return_value.chat.call_args.args
        self.assertEqual(system_prompt, "Você é um robô social assistivo.")
        self.assertIn("Estado de atenção do usuário: ATTENDING", user_prompt)
        self.assertIn("Duração do olhar: 1.2s", user_prompt)
        self.assertIn(
            "Cena observada: Uma pessoa está diante da câmera.", user_prompt
        )


if __name__ == "__main__":
    unittest.main()
