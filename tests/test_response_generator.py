"""Tests for the contextual local-LLM prompt."""

import unittest
from unittest.mock import patch

from src.attention import AttentionState
from src.emotions import EmotionState
from src.llm.response_generator import generate_response


class ResponsePromptTests(unittest.TestCase):
    @patch("src.profiling.record_step")
    @patch("src.llm.response_generator.OllamaClient.from_environment")
    def test_prompt_contains_attention_affect_and_scene(
        self, client_factory, record_step
    ) -> None:
        client_factory.return_value.chat.return_value = "Olá!"
        emotion_state = EmotionState(
            label="focused_expression",
            confidence=0.72,
            blendshapes={"browDownLeft": 0.7},
        )

        result = generate_response(
            "Uma pessoa está diante da câmera.",
            AttentionState.ATTENDING,
            1.24,
            emotion_state=emotion_state,
        )

        self.assertEqual(result, "Olá!")
        system_prompt, user_prompt = client_factory.return_value.chat.call_args.args
        self.assertEqual(
            client_factory.return_value.chat.call_args.kwargs["profiling_name"],
            "qwen_llm",
        )
        self.assertIn("robô social", system_prompt)
        self.assertIn("Não invente intenção, emoção ou identidade.", system_prompt)
        self.assertIn("Estado de atenção do usuário: ATTENDING", user_prompt)
        self.assertIn("Duração do olhar: 1.2s", user_prompt)
        self.assertIn("apparent_affect:", user_prompt)
        self.assertIn("label: focused_expression", user_prompt)
        self.assertIn("confidence: 0.720", user_prompt)
        self.assertIn(
            "Uma pessoa está diante da câmera.", user_prompt
        )


if __name__ == "__main__":
    unittest.main()
