"""Optional real-audio adapter for rendered moktak buffers."""

from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from .display import DEFAULT_MOKTAK_PATH
from .models import MoktakParameters
from .renderer import AudioBuffer, render_moktak, render_moktak_hit


class MoktakPlayer:
    """Play rendered buffers when sounddevice is available, otherwise fall back."""

    def __init__(self, asset_path: Path = DEFAULT_MOKTAK_PATH) -> None:
        self.asset_path = asset_path
        self._samples = np.zeros((0, 1), dtype=np.float32)
        self._sample_rate = 1
        self._hit_samples: AudioBuffer | None = None
        self._hit_sample_rate = 1
        self._started_at = 0.0
        self._beat_clock_started_at: float | None = None
        self.last_error: str | None = None

    def _phase_align(
        self,
        samples: AudioBuffer,
        sample_rate: int,
        parameters: MoktakParameters,
        now: float,
        was_playing: bool,
    ) -> AudioBuffer:
        if self._beat_clock_started_at is None:
            self._beat_clock_started_at = now
            return samples
        if not was_playing:
            return samples

        beat_interval = max(
            1,
            int(round((60.0 / parameters.bpm) * sample_rate)),
        )
        elapsed_frames = int((now - self._beat_clock_started_at) * sample_rate)
        phase_offset = elapsed_frames % beat_interval
        if not phase_offset:
            return samples
        return np.roll(samples, -phase_offset, axis=0)

    def _play_samples(
        self,
        samples: AudioBuffer,
        sample_rate: int,
        *,
        blocking: bool,
        loop: bool,
    ) -> bool:
        try:
            import sounddevice as sd

            sd.stop()
            sd.play(
                samples,
                sample_rate,
                blocking=blocking,
                loop=loop,
            )
            if blocking:
                sd.wait()
            return True
        except Exception as error:
            self.last_error = str(error)
            print(f"moktak audio unavailable; visualizer only: {error}")
            return False

    def play(self, parameters: MoktakParameters) -> bool:
        now = time.monotonic()
        was_playing = bool(len(self._samples))
        samples, sample_rate = render_moktak(
            parameters,
            self.asset_path,
        )
        if parameters.enabled and len(samples):
            samples = self._phase_align(
                samples,
                sample_rate,
                parameters,
                now,
                was_playing,
            )

        self._samples = samples
        self._sample_rate = sample_rate
        self._started_at = now
        self.last_error = None
        if not parameters.enabled or not len(self._samples):
            self._beat_clock_started_at = None
            self.stop()
            return False

        self.last_error = None
        return self._play_samples(
            self._samples,
            self._sample_rate,
            blocking=False,
            loop=True,
        )

    def play_beat(self, gain: float = 1.0) -> bool:
        """Play one natural moktak hit without owning rhythm timing."""
        if self._hit_samples is None:
            self._hit_samples, self._hit_sample_rate = render_moktak_hit(
                self.asset_path,
                gain=1.0,
            )
        samples = np.clip(self._hit_samples * gain, -1.0, 1.0).astype(np.float32)
        sample_rate = self._hit_sample_rate
        self._samples = samples
        self._sample_rate = sample_rate
        self._started_at = time.monotonic()
        self.last_error = None
        if not len(samples):
            return False
        return self._play_samples(samples, sample_rate, blocking=False, loop=False)

    def play_ducked_speech(
        self,
        parameters: MoktakParameters,
        speech_samples: AudioBuffer,
        speech_sample_rate: int,
    ) -> bool:
        """Play generated speech mixed over ducked moktak audio."""
        if not len(speech_samples) or speech_sample_rate <= 0:
            return False

        now = time.monotonic()
        ducked_parameters = MoktakParameters(
            enabled=parameters.enabled,
            bpm=parameters.bpm,
            gain=parameters.gain * parameters.ducking_gain,
            intensity=parameters.intensity,
            regularity=parameters.regularity,
            duration_seconds=parameters.duration_seconds,
            ducking_gain=parameters.ducking_gain,
            fade_in_seconds=parameters.fade_in_seconds,
            fade_out_seconds=parameters.fade_out_seconds,
            frequency_scale=parameters.frequency_scale,
        )
        moktak_samples, moktak_rate = render_moktak(
            ducked_parameters,
            self.asset_path,
        )
        if not len(moktak_samples):
            return self._play_samples(
                speech_samples,
                speech_sample_rate,
                blocking=True,
                loop=False,
            )

        moktak_samples = self._phase_align(
            moktak_samples,
            moktak_rate,
            parameters,
            now,
            bool(len(self._samples)),
        )
        speech = _resample_if_needed(speech_samples, speech_sample_rate, moktak_rate)
        speech = _match_channels(speech, moktak_samples.shape[1])
        indices = np.arange(len(speech)) % len(moktak_samples)
        mixed = np.clip(speech + moktak_samples[indices], -1.0, 1.0).astype(np.float32)

        self._samples = mixed
        self._sample_rate = moktak_rate
        self._started_at = now
        self.last_error = None
        return self._play_samples(mixed, moktak_rate, blocking=True, loop=False)

    def stop(self) -> None:
        try:
            import sounddevice as sd

            sd.stop()
        except Exception:
            pass

    def current_level(self, window_seconds: float = 0.05) -> float:
        if not len(self._samples):
            return 0.0
        start = int(
            (time.monotonic() - self._started_at) * self._sample_rate
        ) % len(self._samples)
        window = max(1, int(window_seconds * self._sample_rate))
        indices = (np.arange(window) + start) % len(self._samples)
        chunk = self._samples[indices]
        return float(np.sqrt(np.mean(np.square(chunk))))

    def waveform(self, max_points: int = 600) -> NDArray[np.float32]:
        if not len(self._samples):
            return np.zeros(0, dtype=np.float32)
        mono = np.mean(self._samples, axis=1)
        if len(mono) <= max_points:
            return mono
        indices = np.linspace(0, len(mono) - 1, max_points).astype(int)
        return mono[indices]

    def __enter__(self) -> "MoktakPlayer":
        return self

    def __exit__(self, *_: object) -> None:
        self.stop()


def _resample_if_needed(
    samples: AudioBuffer,
    source_rate: int,
    target_rate: int,
) -> AudioBuffer:
    if source_rate == target_rate:
        return samples.astype(np.float32)

    gcd = int(np.gcd(source_rate, target_rate))
    up = target_rate // gcd
    down = source_rate // gcd
    return signal.resample_poly(samples, up, down, axis=0).astype(np.float32)


def _match_channels(samples: AudioBuffer, channels: int) -> AudioBuffer:
    if samples.shape[1] == channels:
        return samples.astype(np.float32)
    if channels == 1:
        return np.mean(samples, axis=1, keepdims=True).astype(np.float32)
    if samples.shape[1] == 1:
        return np.repeat(samples, channels, axis=1).astype(np.float32)
    return samples[:, :channels].astype(np.float32)
