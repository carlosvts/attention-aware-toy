"""Temporal filtering for attention scores."""

from enum import IntEnum


class AttentionState(IntEnum):
    """Ordered levels of visual attention detected in the current frame."""

    NO_FACE = 0
    FACE_DETECTED = 1
    LOOKING_BRIEFLY = 2
    DISTRACTED = 3
    ATTENDING = 4

    @classmethod
    def classify(cls, score: float, face_detected: bool) -> "AttentionState":
        """Map face presence and attention score to a discrete state."""
        if not face_detected:
            return cls.NO_FACE
        if score > 0.70:
            return cls.ATTENDING
        if score > 0.50:
            return cls.DISTRACTED
        if score > 0.30:
            return cls.LOOKING_BRIEFLY
        return cls.FACE_DETECTED


class GazeDurationTracker:
    """Measure uninterrupted time spent in the ATTENDING state."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def update(self, state: AttentionState, timestamp: float) -> float:
        """Return current continuous attentive-gaze duration in seconds."""
        if state is not AttentionState.ATTENDING:
            self._started_at = None
            return 0.0
        if self._started_at is None:
            self._started_at = timestamp
        return max(0.0, timestamp - self._started_at)


class SustainedAttentionTracker:
    """Detect when a score stays above a threshold for a minimum duration."""

    def __init__(self, threshold: float = 0.70, minimum_duration: float = 1) -> None:
        self.threshold = threshold
        self.minimum_duration = minimum_duration
        self._attention_started_at: float | None = None

    def update(self, score: float, timestamp: float) -> bool:
        """Return whether attention has been sustained long enough."""
        if score <= self.threshold:
            self._attention_started_at = None
            return False

        if self._attention_started_at is None:
            self._attention_started_at = timestamp

        return timestamp - self._attention_started_at >= self.minimum_duration


class AttentionSessionGate:
    """Allow one interaction until attention has been absent long enough."""

    def __init__(self, threshold: float = 0.7, release_duration: float = 1.0) -> None:
        self.threshold = threshold
        self.release_duration = release_duration
        self._handled = False
        self._attention_lost_at: float | None = None

    def update(
        self,
        score: float,
        timestamp: float,
        sustained_attention: bool,
    ) -> bool:
        """Return true once for each distinct sustained-attention session."""
        if score > self.threshold:
            self._attention_lost_at = None
        elif self._attention_lost_at is None:
            self._attention_lost_at = timestamp
        elif timestamp - self._attention_lost_at >= self.release_duration:
            self._handled = False

        if sustained_attention and not self._handled:
            self._handled = True
            return True
        return False
