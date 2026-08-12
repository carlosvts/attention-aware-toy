"""Pure policy that maps interaction state to moktak playback parameters."""

from typing import Protocol

from .models import MoktakParameters


class LabeledState(Protocol):
    """Minimum interface required from emotion and mudra results."""

    label: str


class NamedState(Protocol):
    """Minimum interface required from an attention state enum."""

    name: str


# TODO: mocked names, change later
_CALM_MUDRA_MARKERS = (
    "calm",
    "calma",
    "meditat",
    "prayer",
    "anjali",
    "open_palm",
)


_BASE_BPM = 60.0
_BASE_GAIN = 0.40
_BASE_INTENSITY = 3.0 / 7.0
_BASE_REGULARITY = 1.0
_BASE_DURATION_SECONDS = 3.0
_BASE_DUCKING_GAIN = 0.40
_NEGATIVE_BPM = 48.0
_CALM_NEGATIVE_BPM = 42.0
_NEUTRAL_BPM = _BASE_BPM
_POSITIVE_BPM = 78.0


def _is_calm_mudra(mudra_state: LabeledState | None) -> bool:
    if mudra_state is None:
        return False
    normalized_label = mudra_state.label.strip().lower().replace("-", "_")
    return any(marker in normalized_label for marker in _CALM_MUDRA_MARKERS)


def _attention_gain(base_gain: float, attention_state: NamedState | None) -> float:
    if attention_state is None:
        return base_gain
    if attention_state.name == "DISTRACTED":
        return base_gain * 0.85
    return base_gain


def _enabled_parameters(
    bpm: float,
    attention_state: NamedState | None,
) -> MoktakParameters:
    """Return moktak parameters editing beat density and attention state."""
    return MoktakParameters(
        enabled=True,
        bpm=bpm,
        gain=_attention_gain(_BASE_GAIN, attention_state),
        intensity=_BASE_INTENSITY,
        regularity=_BASE_REGULARITY,
        duration_seconds=_BASE_DURATION_SECONDS,
        ducking_gain=_BASE_DUCKING_GAIN,
        fade_in_seconds=0.0,
        fade_out_seconds=0.0,
    )


def decide_moktak(
    emotion_state: LabeledState | None,
    mudra_state: LabeledState | None = None,
    llm_response: str = "",
    attention_state: NamedState | None = None,
) -> MoktakParameters:
    """Return a deterministic moktak decision without performing I/O."""
    if attention_state is not None and attention_state.name == "NO_FACE":
        return MoktakParameters.disabled()

    emotion_label = emotion_state.label if emotion_state is not None else ""

    if emotion_label == "negative_expression":
        if _is_calm_mudra(mudra_state):
            # A calm mudra makes the negative-expression cue more spacious.
            return _enabled_parameters(
                _CALM_NEGATIVE_BPM,
                attention_state,
            )
        return _enabled_parameters(
            _NEGATIVE_BPM,
            attention_state,
        )

    if emotion_label == "positive_expression":
        return _enabled_parameters(
            _POSITIVE_BPM,
            attention_state,
        )

    # Neutral and temporarily missing detections keep the base rhythm running.
    return _enabled_parameters(
        _NEUTRAL_BPM,
        attention_state,
    )
