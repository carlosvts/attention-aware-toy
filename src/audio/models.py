"""Data models shared by the moktak audio components."""

from dataclasses import dataclass


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
    # defines frequency scale for the moktak to be played
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
