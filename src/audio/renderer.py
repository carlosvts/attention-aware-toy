"""Render a moktak WAV into a parameterized, in-memory beat pattern."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from .display import DEFAULT_MOKTAK_PATH
from .models import MoktakParameters


AudioBuffer = NDArray[np.float32]
_ENVELOPE_WINDOW_SECONDS = 0.010
_ONSET_THRESHOLD_RATIO = 0.25
_MIN_ONSET_GAP_SECONDS = 0.18
_HIT_PRE_ROLL_SECONDS = 0.005
_HIT_TAIL_SECONDS = 0.18
_HIT_MAX_BEAT_FRACTION = 0.90
_HIT_FADE_OUT_SECONDS = 0.040


def load_wav(path: Path = DEFAULT_MOKTAK_PATH) -> tuple[AudioBuffer, int]:
    """Load a clean WAV as normalized floating-point samples."""
    sample_rate, samples = wavfile.read(path)
    if sample_rate <= 0:
        raise ValueError("moktak WAV must declare a positive sample rate")
    if samples.ndim not in (1, 2):
        raise ValueError("moktak renderer supports mono or stereo WAV files only")
    if samples.size == 0:
        raise ValueError("moktak WAV must contain at least one frame")

    # scipy.io.wavfile gives decoded PCM/float arrays without manual byte parsing.
    if np.issubdtype(samples.dtype, np.integer):
        dtype_info = np.iinfo(samples.dtype)
        scale = float(max(abs(dtype_info.min), dtype_info.max))
        normalized = samples.astype(np.float32) / scale
    elif np.issubdtype(samples.dtype, np.floating):
        normalized = samples.astype(np.float32)
        if np.any((normalized < -1.0) | (normalized > 1.0)):
            raise ValueError("floating-point moktak WAV must be normalized")
    else:
        raise ValueError(f"unsupported moktak WAV dtype: {samples.dtype}")

    if not np.all(np.isfinite(normalized)):
        raise ValueError("moktak WAV contains non-finite samples")
    if normalized.ndim == 1:
        normalized = normalized[:, None]

    return normalized.astype(np.float32), int(sample_rate)


def _shape_intensity(samples: AudioBuffer, intensity: float) -> AudioBuffer:
    """Change the transient body while preserving sample polarity."""
    exponent = 1.15 - 0.35 * intensity
    return np.sign(samples) * np.power(np.abs(samples), exponent)


def _rms_envelope(samples: AudioBuffer, sample_rate: int) -> NDArray[np.float32]:
    mono = np.max(np.abs(samples), axis=1).astype(np.float32)
    window = max(1, int(round(_ENVELOPE_WINDOW_SECONDS * sample_rate)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.sqrt(np.convolve(mono * mono, kernel, mode="same")).astype(np.float32)


def _active_regions(
    envelope: NDArray[np.float32],
    sample_rate: int,
) -> list[tuple[int, int]]:
    peak = float(np.max(envelope)) if len(envelope) else 0.0
    if peak <= 0.0:
        return []

    active = envelope > peak * _ONSET_THRESHOLD_RATIO
    edges = np.diff(active.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if active[0]:
        starts.insert(0, 0)
    if active[-1]:
        ends.append(len(active))

    min_gap = max(1, int(round(_MIN_ONSET_GAP_SECONDS * sample_rate)))
    regions: list[tuple[int, int]] = []
    for start, end in zip(starts, ends):
        if not regions or start - regions[-1][1] > min_gap:
            regions.append((int(start), int(end)))
        else:
            regions[-1] = (regions[-1][0], int(end))
    return regions


def _prepare_hit(
    source: AudioBuffer,
    sample_rate: int,
    beat_interval_frames: int,
) -> AudioBuffer:
    """Extract one natural-speed hit from a source that may contain several hits."""
    regions = _active_regions(_rms_envelope(source, sample_rate), sample_rate)
    if not regions:
        hit = source.copy()
    else:
        first_start, first_end = regions[0]
        pre_roll = int(round(_HIT_PRE_ROLL_SECONDS * sample_rate))
        tail = int(round(_HIT_TAIL_SECONDS * sample_rate))
        start = max(0, first_start - pre_roll)
        natural_end = min(len(source), first_end + tail)
        if len(regions) > 1:
            natural_end = min(natural_end, regions[1][0])
        hit = source[start:max(start + 1, natural_end)].copy()

    max_hit_frames = max(1, int(round(beat_interval_frames * _HIT_MAX_BEAT_FRACTION)))
    if len(hit) > max_hit_frames:
        hit = hit[:max_hit_frames].copy()

    fade_frames = min(len(hit), int(round(_HIT_FADE_OUT_SECONDS * sample_rate)))
    if fade_frames > 1:
        hit[-fade_frames:] *= np.linspace(1.0, 0.0, fade_frames, dtype=np.float32)[
            :, None
        ]
    return hit.astype(np.float32)


def _render_beat_pattern(
    source: AudioBuffer,
    sample_rate: int,
    parameters: MoktakParameters,
    frame_count: int,
) -> AudioBuffer:
    """Place natural-speed moktak hits according to bpm."""
    rendered = np.zeros((frame_count, source.shape[1]), dtype=np.float32)
    beat_interval = max(1, int(round((60.0 / parameters.bpm) * sample_rate)))
    shaped_hit = _shape_intensity(
        _prepare_hit(source, sample_rate, beat_interval),
        parameters.intensity,
    )

    for start in range(0, frame_count, beat_interval):
        end = min(frame_count, start + len(shaped_hit))
        hit_frames = end - start
        if hit_frames <= 0:
            break
        rendered[start:end] += shaped_hit[:hit_frames]

    return rendered


def _beat_interval_frames(parameters: MoktakParameters, sample_rate: int) -> int:
    return max(1, int(round((60.0 / parameters.bpm) * sample_rate)))


def _loop_frame_count(parameters: MoktakParameters, sample_rate: int) -> int:
    requested_frames = max(1, int(parameters.duration_seconds * sample_rate))
    beat_interval = _beat_interval_frames(parameters, sample_rate)
    beat_count = max(1, int(np.ceil(requested_frames / beat_interval)))
    return beat_count * beat_interval


def render_moktak(
    parameters: MoktakParameters,
    path: Path = DEFAULT_MOKTAK_PATH,
) -> tuple[AudioBuffer, int]:
    """Build the exact buffer heard by the player from policy parameters."""
    source, sample_rate = load_wav(path)
    if not parameters.enabled or parameters.duration_seconds == 0.0:
        return np.zeros((0, source.shape[1]), dtype=np.float32), sample_rate

    frame_count = _loop_frame_count(parameters, sample_rate)
    rendered = _render_beat_pattern(source, sample_rate, parameters, frame_count)
    rendered *= parameters.gain

    fade_in_frames = min(frame_count, int(parameters.fade_in_seconds * sample_rate))
    if fade_in_frames:
        rendered[:fade_in_frames] *= np.linspace(
            0.0, 1.0, fade_in_frames, dtype=np.float32
        )[:, None]

    fade_out_frames = min(frame_count, int(parameters.fade_out_seconds * sample_rate))
    if fade_out_frames:
        rendered[-fade_out_frames:] *= np.linspace(
            1.0, 0.0, fade_out_frames, dtype=np.float32
        )[:, None]

    return np.clip(rendered, -1.0, 1.0).astype(np.float32), sample_rate


def render_moktak_hit(
    path: Path = DEFAULT_MOKTAK_PATH,
    *,
    bpm: float = 60.0,
    gain: float = 1.0,
    intensity: float = 3.0 / 7.0,
) -> tuple[AudioBuffer, int]:
    """Render one natural-speed moktak hit for event-driven scheduling."""
    if bpm <= 0.0:
        raise ValueError("bpm must be positive")
    if not 0.0 <= gain <= 1.0:
        raise ValueError("gain must be between 0.0 and 1.0")
    if not 0.0 <= intensity <= 1.0:
        raise ValueError("intensity must be between 0.0 and 1.0")

    source, sample_rate = load_wav(path)
    beat_interval = max(1, int(round((60.0 / bpm) * sample_rate)))
    hit = _shape_intensity(
        _prepare_hit(source, sample_rate, beat_interval),
        intensity,
    )
    return np.clip(hit * gain, -1.0, 1.0).astype(np.float32), sample_rate
