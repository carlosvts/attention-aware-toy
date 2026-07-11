"""Data models shared by the moktak audio components."""

from dataclasses import dataclass
from enum import Enum
import math


@dataclass(frozen=True)
class MoktakParameters:
    """Playback intent produced by the policy, independent of any player."""

    enabled: bool
    bpm: float
    gain: float
    intensity: float
    regularity: float
    duration_seconds: float
    ducking_gain: float
    fade_in_seconds: float
    fade_out_seconds: float
    # Kept for compatibility; the renderer preserves the moktak's natural pitch.
    frequency_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in ("gain", "intensity", "regularity", "ducking_gain"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")

        for name in (
            "bpm",
            "duration_seconds",
            "fade_in_seconds",
            "fade_out_seconds",
        ):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        if self.enabled and self.bpm == 0.0:
            raise ValueError("enabled moktak parameters require a positive bpm")
        if self.frequency_scale <= 0.0:
            raise ValueError("frequency_scale must be positive")
        if self.fade_in_seconds > self.duration_seconds:
            raise ValueError("fade_in_seconds cannot exceed duration_seconds")
        if self.fade_out_seconds > self.duration_seconds:
            raise ValueError("fade_out_seconds cannot exceed duration_seconds")

    @classmethod
    def disabled(cls) -> "MoktakParameters":
        """Return a valid no-playback decision."""
        return cls(
            enabled=False,
            bpm=0.0,
            gain=0.0,
            intensity=0.0,
            regularity=0.0,
            duration_seconds=0.0,
            ducking_gain=1.0,
            fade_in_seconds=0.0,
            fade_out_seconds=0.0,
        )


@dataclass(frozen=True)
class BeatStep:
    """One scheduled moktak hit."""

    interval_seconds: float
    gain: float

    def __post_init__(self) -> None:
        if self.interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be positive")
        if not 0.0 <= self.gain <= 1.0:
            raise ValueError("gain must be between 0.0 and 1.0")


@dataclass(frozen=True)
class MoktakEffectProfile:
    """Adjustable timing profile for the final positive moktak effect."""

    normal_bpm: float = 60.0
    peak_bpm: float = 180.0
    positive_hold_seconds: float = 3.0
    acceleration_steps: int = 5
    deceleration_steps: int = 8
    normal_gain: float = 0.40
    minimum_gain: float = 0.08
    fade_power: float = 1.25
    final_interval_multiplier: float = 1.8

    def __post_init__(self) -> None:
        if self.normal_bpm <= 0.0:
            raise ValueError("normal_bpm must be positive")
        if self.peak_bpm <= self.normal_bpm:
            raise ValueError("peak_bpm must be greater than normal_bpm")
        if self.positive_hold_seconds < 0.0:
            raise ValueError("positive_hold_seconds must be non-negative")
        if self.acceleration_steps < 2:
            raise ValueError("acceleration_steps must be at least 2")
        if self.deceleration_steps < 2:
            raise ValueError("deceleration_steps must be at least 2")
        for name in ("normal_gain", "minimum_gain"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0.0 and 1.0")
        if self.minimum_gain > self.normal_gain:
            raise ValueError("minimum_gain cannot exceed normal_gain")
        if self.fade_power <= 0.0:
            raise ValueError("fade_power must be positive")
        if self.final_interval_multiplier <= 1.0:
            raise ValueError("final_interval_multiplier must be greater than 1.0")

    @property
    def normal_interval_seconds(self) -> float:
        return 60.0 / self.normal_bpm

    def _acceleration_envelope(self, index: int, last_index: int) -> float:
        ratio = index / last_index
        return math.sin(math.pi * ratio / 2.0) ** 2.0

    def _deceleration_envelope(self, index: int, last_index: int) -> float:
        ratio = index / last_index
        return 0.5 * (1.0 + math.cos(math.pi * ratio))

    def positive_steps(self) -> tuple[BeatStep, ...]:
        """Return piecewise sine/cosine acceleration and fade-out deceleration."""
        normal_interval = self.normal_interval_seconds
        peak_interval = 60.0 / self.peak_bpm
        final_interval = normal_interval * self.final_interval_multiplier
        final_bpm = 60.0 / final_interval
        acceleration = [
            BeatStep(
                interval_seconds=60.0
                / (
                    self.normal_bpm
                    + (self.peak_bpm - self.normal_bpm)
                    * self._acceleration_envelope(
                        index,
                        self.acceleration_steps - 1,
                    )
                ),
                gain=self.normal_gain,
            )
            for index in range(self.acceleration_steps)
        ]
        deceleration = [
            BeatStep(
                interval_seconds=60.0
                / (
                    final_bpm
                    + (self.peak_bpm - final_bpm)
                    * self._deceleration_envelope(
                        index,
                        self.deceleration_steps - 1,
                    )
                ),
                gain=self.minimum_gain
                + (self.normal_gain - self.minimum_gain)
                * (
                    self._deceleration_envelope(
                        index,
                        self.deceleration_steps - 1,
                    )
                    ** self.fade_power
                ),
            )
            for index in range(1, self.deceleration_steps)
        ]
        return tuple(acceleration + deceleration)


class MoktakState(Enum):
    """Lifecycle state owned by the moktak audio thread."""

    STOPPED = "stopped"
    PLAYING = "playing"
    POSITIVE_EFFECT = "positive_effect"
    SHUTDOWN = "shutdown"
