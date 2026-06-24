"""Apparent facial-expression detection."""

from .detector import EmotionDetector, classify_expression
from .types import EmotionState

__all__ = ["EmotionDetector", "EmotionState", "classify_expression"]
