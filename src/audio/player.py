"""Optional real-audio adapter for rendered moktak buffers."""

from pathlib import Path
import time

import numpy as np
from numpy.typing import NDArray

from .display import DEFAULT_MOKTAK_PATH
from .models import MoktakParameters
from .renderer import AudioBuffer, render_moktak


class MoktakPlayer:
    """Play rendered buffers when sounddevice is available, otherwise fall back."""

    def __init__(self, asset_path: Path = DEFAULT_MOKTAK_PATH) -> None:
        self.asset_path = asset_path
        self._samples = np.zeros((0, 1), dtype=np.float32)
        self._sample_rate = 1
        self._started_at = 0.0
        self._beat_clock_started_at: float | None = None
        self.last_error: str | None = None

    def play(self, parameters: MoktakParameters) -> bool:
        now = time.monotonic()
        was_playing = bool(len(self._samples))
        samples, sample_rate = render_moktak(
            parameters,
            self.asset_path,
        )
        if parameters.enabled and len(samples):
            if self._beat_clock_started_at is None:
                self._beat_clock_started_at = now
            elif was_playing:
                beat_interval = max(
                    1,
                    int(round((60.0 / parameters.bpm) * sample_rate)),
                )
                elapsed_frames = int((now - self._beat_clock_started_at) * sample_rate)
                phase_offset = elapsed_frames % beat_interval
                if phase_offset:
                    samples = np.roll(samples, -phase_offset, axis=0)

        self._samples = samples
        self._sample_rate = sample_rate
        self._started_at = now
        self.last_error = None
        if not parameters.enabled or not len(self._samples):
            self._beat_clock_started_at = None
            self.stop()
            return False

        try:
            import sounddevice as sd

            sd.stop()
            sd.play(
                self._samples,
                self._sample_rate,
                blocking=False,
                loop=True,
            )
            return True
        except Exception as error:
            self.last_error = str(error)
            print(f"moktak audio unavailable; visualizer only: {error}")
            return False

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
