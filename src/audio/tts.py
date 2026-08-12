"""Optional text-to-speech adapter for spoken meditation guidance."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile

import numpy as np
from numpy.typing import NDArray


SpeechBuffer = NDArray[np.float32]

_DEFAULT_TTS_MODEL = "tts-1"
_DEFAULT_TTS_VOICE = "alloy"
_DEFAULT_TTS_INSTRUCTIONS = (
    "Speak slowly and calmly, like a grounded Buddhist meditation guide. "
    "Use a warm, spacious tone with gentle pacing."
)
_DEFAULT_TTS_PROVIDER = "generic"


@dataclass(frozen=True)
class SpeechAudio:
    samples: SpeechBuffer
    sample_rate: int


class OpenAITTS:
    """Generate speech audio when OpenAI credentials and SDK are available."""

    def __init__(
        self,
        model: str | None = None,
        voice: str | None = None,
        instructions: str | None = None,
    ) -> None:
        self.model = model or os.getenv("OPENAI_TTS_MODEL", _DEFAULT_TTS_MODEL)
        self.voice = voice or os.getenv("OPENAI_TTS_VOICE", _DEFAULT_TTS_VOICE)
        self.instructions = (
            instructions
            or os.getenv("OPENAI_TTS_INSTRUCTIONS")
            or _DEFAULT_TTS_INSTRUCTIONS
        )
        self.last_error: str | None = None

    def synthesize(self, text: str) -> SpeechAudio | None:
        """Return generated speech, or None when TTS is unavailable."""
        if not text.strip():
            return None
        if not os.getenv("OPENAI_API_KEY"):
            self.last_error = "OPENAI_API_KEY is not set"
            return None

        try:
            import soundfile as sf
            from openai import OpenAI
        except Exception as error:
            self.last_error = f"TTS dependencies unavailable: {error}"
            return None

        output_path: Path | None = None
        try:
            client = OpenAI()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output:
                output_path = Path(output.name)

            speech_args = {
                "model": self.model,
                "voice": self.voice,
                "input": text,
                "response_format": "wav",
            }
            if self.model not in {"tts-1", "tts-1-hd"}:
                speech_args["instructions"] = self.instructions

            with client.audio.speech.with_streaming_response.create(
                **speech_args,
            ) as response:
                response.stream_to_file(output_path)

            samples, sample_rate = sf.read(
                output_path,
                dtype="float32",
                always_2d=True,
            )
            if sample_rate <= 0 or not len(samples):
                self.last_error = "OpenAI TTS returned empty audio"
                return None

            self.last_error = None
            return SpeechAudio(
                samples=np.asarray(samples, dtype=np.float32),
                sample_rate=int(sample_rate),
            )
        except Exception as error:
            self.last_error = f"OpenAI TTS failed: {error}"
            return None
        finally:
            if output_path is not None:
                try:
                    output_path.unlink(missing_ok=True)
                except Exception:
                    pass


class GenericTTS:
    """Placeholder TTS provider that avoids external API calls."""

    def __init__(self) -> None:
        self.last_error: str | None = None

    def synthesize(self, text: str) -> SpeechAudio | None:
        if not text.strip():
            return None
        self.last_error = "Generic TTS placeholder; speech is printed only"
        return None


def create_tts():
    """Return the configured TTS provider."""
    provider = os.getenv("TTS_PROVIDER", _DEFAULT_TTS_PROVIDER).strip().lower()
    if provider == "openai":
        return OpenAITTS()
    return GenericTTS()
