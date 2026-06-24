"""Simple local mocks for modules that are not available yet."""

from dataclasses import dataclass

from numpy.typing import NDArray

from src.emotions import EmotionState


@dataclass(frozen=True)
class MudraState:
    label: str
    confidence: float


class MudraDetectorMock:
    def detect(self, frame_bgr: NDArray) -> MudraState:
        return MudraState(label="mock_mudra", confidence=1.0)


class GestureDescriptionMock:
    def describe(
        self,
        mudra_state: MudraState,
        emotion_state: EmotionState | None,
    ) -> str:
        label = emotion_state.label if emotion_state else "unknown"
        return (
            f"Mock description: mudra={mudra_state.label}, "
            f"apparent_expression={label}"
        )


class LLMResponseMock:
    def generate(self, gesture_description: str) -> str:
        return f"Mock LLM response: {gesture_description}"
