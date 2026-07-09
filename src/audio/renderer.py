"""Render a moktak WAV into a parameterized, in-memory beat pattern."""

from pathlib import Path
import wave

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .display import DEFAULT_MOKTAK_PATH
from .models import MoktakParameters


AudioBuffer = NDArray[np.float32]


def load_wav(path: Path = DEFAULT_MOKTAK_PATH) -> tuple[AudioBuffer, int]:
    """Load the bundled 16-bit PCM WAV as normalized floating-point samples."""
    with wave.open(str(path), "rb") as wav_file:
        if wav_file.getsampwidth() != 2:
            raise ValueError("moktak renderer supports 16-bit PCM WAV files only")
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        frames = wav_file.readframes(wav_file.getnframes())
    # normalize it
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    return samples.reshape(-1, channels), sample_rate


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


def _phase_vocoder(
    spectrum: NDArray[np.complex64] | NDArray[np.complex128],
    rate: float,
    hop_length: int,
    fft_size: int,
) -> NDArray[np.complex64]:
    """Time-stretch an STFT matrix while preserving pitch."""
    if spectrum.shape[1] < 2:
        return spectrum.astype(np.complex64)

    time_steps = np.arange(0, spectrum.shape[1] - 1, rate, dtype=np.float64)
    stretched = np.zeros(
        (spectrum.shape[0], len(time_steps)),
        dtype=np.complex64,
    )
    phase_advance = (
        2.0
        * np.pi
        * hop_length
        * np.arange(spectrum.shape[0], dtype=np.float64)
        / fft_size
    )
    phase = np.angle(spectrum[:, 0]).astype(np.float64)

    for index, step in enumerate(time_steps):
        left = int(np.floor(step))
        right = min(left + 1, spectrum.shape[1] - 1)
        fraction = step - left

        # Interpolate magnitudes, then advance phase coherently to avoid warble.
        magnitude = (
            (1.0 - fraction) * np.abs(spectrum[:, left])
            + fraction * np.abs(spectrum[:, right])
        )
        stretched[:, index] = magnitude * np.exp(1j * phase)

        delta = np.angle(spectrum[:, right]) - np.angle(spectrum[:, left])
        delta -= phase_advance
        delta -= 2.0 * np.pi * np.round(delta / (2.0 * np.pi))
        phase += phase_advance + delta

    return stretched


def _time_stretch_channel(
    channel: NDArray[np.float32],
    rate: float,
) -> NDArray[np.float32]:
    """Stretch duration without changing pitch using a small phase vocoder."""
    if np.isclose(rate, 1.0) or len(channel) < 32:
        return channel.copy()

    # STFT size follows the source length so short test tones remain valid.
    fft_size = min(2048, 2 ** int(np.floor(np.log2(len(channel)))))
    fft_size = max(32, fft_size)
    hop_length = max(1, fft_size // 4)

    _, _, spectrum = signal.stft(
        channel,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_length,
        boundary="zeros",
        padded=True,
    )
    stretched_spectrum = _phase_vocoder(
        spectrum,
        rate,
        hop_length,
        fft_size,
    )
    _, stretched = signal.istft(
        stretched_spectrum,
        window="hann",
        nperseg=fft_size,
        noverlap=fft_size - hop_length,
        input_onesided=True,
        boundary=True,
    )

    target_length = max(1, int(round(len(channel) / rate)))
    if len(stretched) < target_length:
        stretched = np.pad(stretched, (0, target_length - len(stretched)))
    return stretched[:target_length].astype(np.float32)


def _shift_frequency(samples: AudioBuffer, frequency_scale: float) -> AudioBuffer:
    """Raise or lower the perceived frequency while preserving duration."""
    if np.isclose(frequency_scale, 1.0):
        return samples.copy()

    # Pitch shift = stretch without pitch change, then resample back to length.
    stretch_rate = 1.0 / frequency_scale
    shifted_channels = []
    for channel in range(samples.shape[1]):
        stretched = _time_stretch_channel(samples[:, channel], stretch_rate)
        shifted = signal.resample(stretched, len(samples)).astype(np.float32)
        shifted_channels.append(shifted)

    return np.column_stack(shifted_channels).astype(np.float32)


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
    rendered = _shift_frequency(source, parameters.frequency_scale)
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
