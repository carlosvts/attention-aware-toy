"""Tests for emotion-driven moktak audio rendering."""

from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np

from src.audio import MoktakParameters, decide_moktak, render_moktak


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    samples = np.asarray(samples, dtype=np.float32)
    pcm = np.clip(samples, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    if pcm.ndim == 1:
        channels = 1
    else:
        channels = pcm.shape[1]

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def _parameters(
    *,
    duration_seconds: float,
    bpm: float = 60.0,
    frequency_scale: float = 1.0,
) -> MoktakParameters:
    return MoktakParameters(
        enabled=True,
        bpm=bpm,
        gain=1.0,
        intensity=3.0 / 7.0,
        regularity=1.0,
        duration_seconds=duration_seconds,
        ducking_gain=0.40,
        fade_in_seconds=0.0,
        fade_out_seconds=0.0,
        frequency_scale=frequency_scale,
    )


def _dominant_frequency(samples: np.ndarray, sample_rate: int) -> float:
    mono = np.mean(samples, axis=1)
    spectrum = np.abs(np.fft.rfft(mono))
    frequencies = np.fft.rfftfreq(len(mono), 1.0 / sample_rate)
    return float(frequencies[int(np.argmax(spectrum[1:]) + 1)])


def test_policy_changes_timing_instead_of_frequency() -> None:
    negative = decide_moktak(SimpleNamespace(label="negative_expression"))
    neutral = decide_moktak(SimpleNamespace(label="neutral_expression"))
    positive = decide_moktak(SimpleNamespace(label="positive_expression"))

    assert (
        negative.duration_seconds
        == neutral.duration_seconds
        == positive.duration_seconds
    )
    assert negative.bpm < neutral.bpm < positive.bpm
    assert negative.intensity == neutral.intensity == positive.intensity
    assert negative.frequency_scale == neutral.frequency_scale == positive.frequency_scale


def test_renderer_spaces_hits_by_bpm_without_resampling_source(tmp_path: Path) -> None:
    sample_rate = 1000
    source = np.zeros(sample_rate, dtype=np.float32)
    source[100] = 1.0
    source[500] = 1.0
    wav_path = tmp_path / "pulse.wav"
    _write_wav(wav_path, source, sample_rate)

    rendered, rendered_rate = render_moktak(
        _parameters(duration_seconds=2.0),
        wav_path,
    )

    assert rendered_rate == sample_rate
    assert len(rendered) == 2 * sample_rate
    hits = np.flatnonzero(rendered[:, 0] > 0.95)
    assert hits.tolist() == [9, 1009]

    faster, _ = render_moktak(
        _parameters(duration_seconds=2.0, bpm=120.0),
        wav_path,
    )
    faster_hits = np.flatnonzero(faster[:, 0] > 0.95)
    assert faster_hits.tolist() == [9, 509, 1009, 1509]


def test_renderer_preserves_source_frequency(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_seconds = 1.0
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    source = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    stereo = np.column_stack((source, source))
    wav_path = tmp_path / "tone.wav"
    _write_wav(wav_path, stereo, sample_rate)

    base, _ = render_moktak(
        _parameters(duration_seconds=duration_seconds, frequency_scale=1.0),
        wav_path,
    )
    lower, _ = render_moktak(
        _parameters(duration_seconds=duration_seconds, frequency_scale=0.80),
        wav_path,
    )
    higher, _ = render_moktak(
        _parameters(duration_seconds=duration_seconds, frequency_scale=1.25),
        wav_path,
    )

    assert len(lower) == len(base) == len(higher) == sample_rate
    base_frequency = _dominant_frequency(base, sample_rate)
    assert np.isclose(_dominant_frequency(lower, sample_rate), base_frequency)
    assert np.isclose(_dominant_frequency(higher, sample_rate), base_frequency)
