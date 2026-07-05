"""OpenCV drawing and debug-window helpers."""

import cv2
import numpy as np
from numpy.typing import NDArray

from src.attention import AttentionResult, AttentionState
from src.emotions import EmotionState


def show_or_close(name: str, enabled: bool, image: NDArray[np.uint8]) -> None:
    if enabled:
        cv2.imshow(name, image)
    else:
        try:
            cv2.destroyWindow(name)
        except cv2.error:
            pass


def draw_attention_overlay(
    frame: NDArray[np.uint8],
    result: AttentionResult,
    attention_state: AttentionState,
    gaze_duration: float,
) -> None:
    color = {
        AttentionState.NO_FACE: (120, 120, 120),
        AttentionState.FACE_DETECTED: (0, 180, 255),
        AttentionState.LOOKING_BRIEFLY: (0, 220, 220),
        AttentionState.DISTRACTED: (0, 140, 255),
        AttentionState.ATTENDING: (80, 220, 80),
    }[attention_state]

    if result.face_box is not None:
        x, y, width, height = result.face_box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)

    label = f"state: {attention_state.name}  score: {result.score:.2f}"
    cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    cv2.putText(
        frame,
        f"gaze duration: {gaze_duration:.1f}s",
        (20, 68),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        color,
        2,
        cv2.LINE_AA,
    )

    if result.features is not None:
        for contour in (result.features.right_eye, result.features.left_eye):
            cv2.polylines(
                frame,
                [np.asarray(contour, dtype=np.int32)],
                True,
                (80, 220, 80),
                1,
                cv2.LINE_AA,
            )
        for center, radius in (result.features.right_iris, result.features.left_iris):
            cv2.circle(frame, center, radius, (255, 80, 220), 2, cv2.LINE_AA)


def draw_emotion_overlay(
    frame: NDArray[np.uint8],
    emotion: EmotionState | None,
    origin: tuple[int, int] = (20, 35),
) -> None:
    text = (
        "apparent_expression=none"
        if emotion is None
        else f"apparent_expression={emotion.label}\nconfidence={emotion.confidence:.2f})"
    )
    cv2.putText(
        frame,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (120, 220, 255),
        2,
        cv2.LINE_AA,
    )


def render_blendshapes_debug(
    emotion: EmotionState | None,
    top_n: int = 15,
    size: tuple[int, int] = (540, 680),
) -> NDArray[np.uint8]:
    height, width = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Blendshapes Debug",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (150, 220, 150),
        1,
        cv2.LINE_AA,
    )
    if emotion is None:
        cv2.putText(
            canvas,
            "No face detected",
            (18, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (120, 120, 120),
            1,
            cv2.LINE_AA,
        )
        return canvas

    items = sorted(emotion.blendshapes.items(), key=lambda item: item[1], reverse=True)
    label_width = 220
    bar_max = width - label_width - 65
    row = 72
    for name, value in items[:top_n]:
        cv2.putText(
            canvas,
            f"{name[:25]:25s}",
            (18, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        x = label_width
        y = row - 14
        cv2.rectangle(canvas, (x, y), (x + bar_max, y + 12), (55, 55, 55), 1)
        cv2.rectangle(
            canvas,
            (x, y),
            (x + int(max(0.0, min(1.0, value)) * bar_max), y + 12),
            (80, 180, 240),
            -1,
        )
        cv2.putText(
            canvas,
            f"{value:.3f}",
            (x + bar_max + 10, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        row += 31
    return canvas


def render_emotion_debug(
    emotion: EmotionState | None,
    top_n: int = 12,
    size: tuple[int, int] = (560, 720),
) -> NDArray[np.uint8]:
    """Render apparent-expression metrics plus top blendshape bars."""
    height, width = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(
        canvas,
        "Emotion Debug",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (120, 220, 255),
        1,
        cv2.LINE_AA,
    )

    if emotion is None:
        lines = (
            "apparent_expression: none",
            "confidence: 0.000",
        )
        color = (120, 120, 120)
    else:
        lines = (
            f"apparent_expression: {emotion.label}",
            f"confidence: {emotion.confidence:.3f}",
        )
        color = (230, 230, 230)

    row = 76
    for line in lines:
        cv2.putText(
            canvas,
            line,
            (18, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            1,
            cv2.LINE_AA,
        )
        row += 30

    cv2.putText(
        canvas,
        "Top blendshapes",
        (18, row + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (150, 220, 150),
        1,
        cv2.LINE_AA,
    )
    row += 60

    if emotion is None:
        return canvas

    items = sorted(emotion.blendshapes.items(), key=lambda item: item[1], reverse=True)
    label_width = 230
    bar_max = width - label_width - 70
    for name, value in items[:top_n]:
        cv2.putText(
            canvas,
            f"{name[:25]:25s}",
            (18, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )
        x = label_width
        y = row - 14
        cv2.rectangle(canvas, (x, y), (x + bar_max, y + 12), (55, 55, 55), 1)
        cv2.rectangle(
            canvas,
            (x, y),
            (x + int(max(0.0, min(1.0, value)) * bar_max), y + 12),
            (80, 180, 240),
            -1,
        )
        cv2.putText(
            canvas,
            f"{value:.3f}",
            (x + bar_max + 10, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        row += 31
    return canvas
