"""Types for apparent facial-expression detection."""

from dataclasses import dataclass


@dataclass(frozen=True)
class EmotionState:
    """Heuristic apparent expression derived from facial blendshapes."""

    label: str
    confidence: float
    blendshapes: dict[str, float]
