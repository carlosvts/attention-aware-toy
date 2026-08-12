"""Tests for emotion-driven moktak audio rendering."""

from pathlib import Path
import math
from types import SimpleNamespace
import wave

import numpy as np

from src.audio import (
    MoktakEffectProfile,
    MoktakParameters,
    decide_moktak,
    render_moktak,
    render_moktak_hit,
)


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


def test_single_hit_renderer_preserves_source_frequency(tmp_path: Path) -> None:
    sample_rate = 8000
    duration_seconds = 0.20
    time = np.arange(int(sample_rate * duration_seconds)) / sample_rate
    source = np.sin(2.0 * np.pi * 440.0 * time).astype(np.float32)
    wav_path = tmp_path / "hit.wav"
    _write_wav(wav_path, source, sample_rate)

    hit, hit_rate = render_moktak_hit(wav_path, bpm=60.0, gain=0.5)

    assert hit_rate == sample_rate
    assert len(hit) <= len(source)
    assert np.isclose(_dominant_frequency(hit, sample_rate), 440.0, atol=5.0)
    assert np.max(np.abs(hit)) <= 0.5


def test_positive_effect_uses_curved_acceleration_and_fade() -> None:
    profile = MoktakEffectProfile(
        normal_bpm=60.0,
        peak_bpm=180.0,
        acceleration_steps=5,
        deceleration_steps=7,
        normal_gain=0.8,
        minimum_gain=0.2,
        fade_power=1.5,
    )

    steps = profile.positive_steps()
    intervals = [step.interval_seconds for step in steps]
    gains = [step.gain for step in steps]
    acceleration = intervals[: profile.acceleration_steps]
    deceleration = intervals[profile.acceleration_steps - 1 :]
    acceleration_deltas = [
        left - right for left, right in zip(acceleration, acceleration[1:])
    ]

    assert all(left > right for left, right in zip(acceleration, acceleration[1:]))
    assert all(left < right for left, right in zip(deceleration, deceleration[1:]))
    assert all(
        left > right
        for left, right in zip(
            gains[profile.acceleration_steps - 1 :],
            gains[profile.acceleration_steps :],
        )
    )
    assert not np.allclose(acceleration_deltas, acceleration_deltas[0])

    midpoint = profile.acceleration_steps // 2
    expected_mid_progress = math.sin(
        math.pi * (midpoint / (profile.acceleration_steps - 1)) / 2.0
    ) ** 2.0
    expected_mid_bpm = profile.normal_bpm + (
        profile.peak_bpm - profile.normal_bpm
    ) * expected_mid_progress
    assert np.isclose(acceleration[midpoint], 60.0 / expected_mid_bpm)

    first_down_ratio = 1 / (profile.deceleration_steps - 1)
    first_down_energy = 0.5 * (1.0 + math.cos(math.pi * first_down_ratio))
    final_bpm = profile.normal_bpm / profile.final_interval_multiplier
    expected_first_down_bpm = final_bpm + (
        profile.peak_bpm - final_bpm
    ) * first_down_energy
    assert np.isclose(intervals[profile.acceleration_steps], 60.0 / expected_first_down_bpm)
