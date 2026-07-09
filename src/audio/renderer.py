"""Render a moktak WAV into a parameterized, in-memory beat pattern."""

from pathlib import Path

import librosa
import numpy as np
from numpy.typing import NDArray
from scipy.io import wavfile

from .display import DEFAULT_MOKTAK_PATH
from .models import MoktakParameters


AudioBuffer = NDArray[np.float32]


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


def _fit_duration(samples: AudioBuffer, frame_count: int) -> AudioBuffer:
    """Fit playback length without changing the source playback speed."""
    if len(samples) == frame_count:
        return samples.copy()

    if len(samples) > frame_count:
        # Cutting keeps the original sample spacing, so playback speed stays fixed.
        return samples[:frame_count].copy()

    # Repeating keeps every hit at its original speed instead of stretching it.
    repeats = int(np.ceil(frame_count / len(samples)))
    return np.tile(samples, (repeats, 1))[:frame_count].astype(np.float32)


def _shift_frequency(
    samples: AudioBuffer,
    sample_rate: int,
    frequency_scale: float,
) -> AudioBuffer:
    """Raise or lower the perceived frequency while preserving duration."""
    if np.isclose(frequency_scale, 1.0):
        return samples.copy()

    # Librosa works in semitone steps: scale 2.0 = +12, scale 0.5 = -12.
    semitones = float(12.0 * np.log2(frequency_scale))
    shifted_channels = [
        librosa.effects.pitch_shift(
            y=samples[:, channel],
            sr=sample_rate,
            n_steps=semitones,
        ).astype(np.float32)
        for channel in range(samples.shape[1])
    ]

    # The algorithm is duration-preserving, but this keeps frame count exact.
    return _fit_duration(
        np.column_stack(shifted_channels).astype(np.float32),
        len(samples),
    )


def render_moktak(
    parameters: MoktakParameters,
    path: Path = DEFAULT_MOKTAK_PATH,
) -> tuple[AudioBuffer, int]:
    """Build the exact buffer heard by the player from policy parameters."""
    source, sample_rate = load_wav(path)
    if not parameters.enabled or parameters.duration_seconds == 0.0:
        return np.zeros((0, source.shape[1]), dtype=np.float32), sample_rate

    frame_count = max(1, int(parameters.duration_seconds * sample_rate))
    # Frequency is changed first; duration fitting later never rescales time.
    rendered = _shift_frequency(source, sample_rate, parameters.frequency_scale)
    rendered = _fit_duration(rendered, frame_count)
    rendered = _shape_intensity(rendered, parameters.intensity)
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
