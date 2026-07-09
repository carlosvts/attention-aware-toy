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
_NEGATIVE_FREQUENCY_SCALE = 0.88
_CALM_NEGATIVE_FREQUENCY_SCALE = 0.82
_NEUTRAL_FREQUENCY_SCALE = 1.0
_POSITIVE_FREQUENCY_SCALE = 1.12


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
    frequency_scale: float,
    attention_state: NamedState | None,
) -> MoktakParameters:
    """Return moktak parameters editing just frequency scale and attention state."""
    # Keep rhythm/body stable across emotions; emotion only changes frequency.
    return MoktakParameters(
        enabled=True,
        bpm=_BASE_BPM,
        gain=_attention_gain(_BASE_GAIN, attention_state),
        intensity=_BASE_INTENSITY,
        regularity=_BASE_REGULARITY,
        duration_seconds=_BASE_DURATION_SECONDS,
        ducking_gain=_BASE_DUCKING_GAIN,
        fade_in_seconds=0.0,
        fade_out_seconds=0.0,
        frequency_scale=frequency_scale,
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
            # A calm mudra makes the negative-expression cue deeper, not slower.
            return _enabled_parameters(
                _CALM_NEGATIVE_FREQUENCY_SCALE,
                attention_state,
            )
        # Negative apparent expression lowers the moktak frequency.
        return _enabled_parameters(
            _NEGATIVE_FREQUENCY_SCALE,
            attention_state,
        )

    if emotion_label == "positive_expression":
        # Positive apparent expression raises the moktak frequency.
        return _enabled_parameters(
            _POSITIVE_FREQUENCY_SCALE,
            attention_state,
        )

    # Neutral and temporarily missing detections keep the base rhythm running.
    return _enabled_parameters(
        _NEUTRAL_FREQUENCY_SCALE,
        attention_state,
    )
