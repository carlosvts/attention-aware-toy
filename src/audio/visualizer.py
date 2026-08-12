"""OpenCV visualization for the buffer produced by the moktak renderer."""

import cv2
import numpy as np
from numpy.typing import NDArray

from .models import MoktakParameters


def render_moktak_visualizer(
    parameters: MoktakParameters,
    waveform: NDArray[np.float32],
    current_level: float,
    status: str = "",
    size: tuple[int, int] = (520, 760),
) -> NDArray[np.uint8]:
    """Return a debug panel representing the actual rendered audio buffer."""
    height, width = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    color = (100, 220, 160) if parameters.enabled else (120, 120, 120)

    cv2.putText(
        canvas,
        "Moktak Audio",
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.82,
        color,
        2,
        cv2.LINE_AA,
    )
    lines = (
        f"enabled: {parameters.enabled}",
        f"bpm: {parameters.bpm:.1f}",
        f"gain: {parameters.gain:.2f}",
        f"intensity: {parameters.intensity:.2f}",
        f"regularity: {parameters.regularity:.2f}",
        f"duration: {parameters.duration_seconds:.2f}s",
    )
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (20, 78 + index * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    bar_left, bar_top, bar_width = 20, 260, width - 40
    cv2.rectangle(
        canvas,
        (bar_left, bar_top),
        (bar_left + bar_width, bar_top + 24),
        (70, 70, 70),
        1,
    )
    normalized_level = min(1.0, max(0.0, current_level * 4.0))
    cv2.rectangle(
        canvas,
        (bar_left, bar_top),
        (bar_left + int(bar_width * normalized_level), bar_top + 24),
        color,
        -1,
    )
    cv2.putText(
        canvas,
        "live output level",
        (bar_left, bar_top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )

    center_y = 380
    if len(waveform):
        x = np.linspace(20, width - 20, len(waveform)).astype(np.int32)
        scale = 90.0 / max(0.05, float(np.max(np.abs(waveform))))
        y = np.clip(center_y - waveform * scale, 300, 460).astype(np.int32)
        points = np.column_stack((x, y))
        cv2.polylines(canvas, [points], False, color, 1, cv2.LINE_AA)
    else:
        cv2.line(canvas, (20, center_y), (width - 20, center_y), color, 1)

    help_text = status or "continuous loop | Q/ESC: exit"
    cv2.putText(
        canvas,
        help_text[:80],
        (20, height - 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (160, 190, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas
