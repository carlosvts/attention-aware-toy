"""Output adapter for moktak decisions.

This first adapter deliberately logs playback intent instead of playing audio.
"""

from pathlib import Path
from typing import TextIO
import sys

from .models import MoktakParameters


DEFAULT_MOKTAK_PATH = Path(__file__).with_name("moktak.wav")


class MoktakDisplay:
    """Render a moktak decision without owning policy or audio playback."""

    def __init__(
        self,
        asset_path: Path = DEFAULT_MOKTAK_PATH,
        stream: TextIO | None = None,
    ) -> None:
        self.asset_path = asset_path
        self.stream = stream

    def show(self, parameters: MoktakParameters) -> None:
        output = self.stream or sys.stdout
        if not parameters.enabled:
            print("moktak: disabled", file=output, flush=True)
            return

        print(
            "moktak: "
            f"asset={self.asset_path} "
            f"bpm={parameters.bpm:.1f} "
            f"gain={parameters.gain:.2f} "
            f"intensity={parameters.intensity:.2f} "
            f"regularity={parameters.regularity:.2f} "
            f"frequency_scale={parameters.frequency_scale:.2f} "
            f"duration={parameters.duration_seconds:.2f}s "
            f"ducking_gain={parameters.ducking_gain:.2f} "
            f"fade_in={parameters.fade_in_seconds:.2f}s "
            f"fade_out={parameters.fade_out_seconds:.2f}s",
            file=output,
            flush=True,
        )
