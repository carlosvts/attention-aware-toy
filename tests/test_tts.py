"""Tests for optional text-to-speech integration."""

import unittest
from unittest.mock import patch

from src.audio.tts import GenericTTS, OpenAITTS, create_tts


class OpenAITTSTests(unittest.TestCase):
    @patch.dict("os.environ", {}, clear=True)
    def test_default_tts_provider_is_generic(self) -> None:
        self.assertIsInstance(create_tts(), GenericTTS)

    def test_generic_tts_does_not_call_external_api(self) -> None:
        tts = GenericTTS()

        self.assertIsNone(tts.synthesize("Breathe in and out."))
        self.assertEqual(
            tts.last_error,
            "Generic TTS placeholder; speech is printed only",
        )

    @patch.dict("os.environ", {}, clear=True)
    def test_synthesize_returns_none_without_api_key(self) -> None:
        tts = OpenAITTS()

        self.assertIsNone(tts.synthesize("Breathe in and out."))
        self.assertEqual(tts.last_error, "OPENAI_API_KEY is not set")


if __name__ == "__main__":
    unittest.main()
