"""Manual webcam debug script for the event-driven moktak logic.

Run from the project root with: python -m tests.test_moktak
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from numpy.typing import NDArray

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app import _handle_moktak_attention_event, _handle_moktak_emotion_event
from src.attention import AttentionDetector, AttentionState, GazeDurationTracker
from src.audio import (
    AttentionActivationTracker,
    MoktakSession,
    MoktakState,
    PositiveEmotionHoldTracker,
    build_effect_profile_from_env,
)
from src.config import env_float, env_int
from src.debug import draw_attention_overlay, draw_emotion_overlay, render_emotion_debug
from src.emotions import EmotionDetector, EmotionState


CAMERA_INDEX = 0
CAMERA_WINDOW_NAME = "Moktak Camera Test"
EMOTION_WINDOW_NAME = "Moktak Emotion Debug"
MOKTAK_WINDOW_NAME = "Moktak Logic Debug"
ATTENTION_DURATION_SECONDS = env_float("ATTENTION_DURATION_SECONDS", 1.0)


class PositiveDurationMeter:
    """Track visible positive-expression duration for the debug window."""

    def __init__(self) -> None:
        self._started_at: float | None = None

    def update(self, emotion: EmotionState | None, timestamp: float) -> float:
        label = emotion.label if emotion is not None else ""
        if label != "positive_expression":
            self._started_at = None
            return 0.0
        if self._started_at is None:
            self._started_at = timestamp
        return max(0.0, timestamp - self._started_at)

    def reset(self) -> None:
        self._started_at = None


def _render_moktak_debug(
    *,
    attention_state: AttentionState,
    gaze_duration: float,
    sustained_attention: bool,
    emotion: EmotionState | None,
    positive_seconds: float,
    positive_hold_seconds: float,
    moktak_state: MoktakState,
    size: tuple[int, int] = (520, 760),
) -> NDArray[np.uint8]:
    height, width = size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    color = {
        MoktakState.STOPPED: (120, 120, 120),
        MoktakState.PLAYING: (80, 220, 120),
        MoktakState.POSITIVE_EFFECT: (80, 180, 255),
        MoktakState.SHUTDOWN: (80, 80, 80),
    }[moktak_state]

    emotion_label = emotion.label if emotion is not None else "none"
    emotion_confidence = emotion.confidence if emotion is not None else 0.0
    sustained_seconds = (
        gaze_duration if attention_state is AttentionState.ATTENDING else 0.0
    )
    lines = (
        "Moktak Logic Debug",
        f"moktak_state: {moktak_state.name}",
        f"attention_state: {attention_state.name}",
        f"sustained_attention: {sustained_attention}",
        f"sustained_attention_seconds: {sustained_seconds:.2f}",
        f"attention_trigger_after: {ATTENTION_DURATION_SECONDS:.2f}s",
        f"emotion: {emotion_label}",
        f"emotion_confidence: {emotion_confidence:.3f}",
        f"positive_expression_seconds: {positive_seconds:.2f}",
        f"positive_effect_after: {positive_hold_seconds:.2f}s",
        "Q/ESC: exit",
    )

    row = 40
    for index, line in enumerate(lines):
        cv2.putText(
            canvas,
            line,
            (22, row),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.78 if index == 0 else 0.58,
            color if index in (0, 1) else (225, 225, 225),
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
        row += 42 if index == 0 else 32

    attention_progress = min(
        1.0,
        sustained_seconds / max(ATTENTION_DURATION_SECONDS, 1e-9),
    )
    positive_progress = min(
        1.0,
        positive_seconds / max(positive_hold_seconds, 1e-9),
    )
    _draw_progress_bar(
        canvas,
        label="attention activation",
        value=attention_progress,
        top=height - 116,
        color=(80, 220, 120),
    )
    _draw_progress_bar(
        canvas,
        label="positive effect",
        value=positive_progress,
        top=height - 62,
        color=(80, 180, 255),
    )
    return canvas


def _draw_progress_bar(
    canvas: NDArray[np.uint8],
    *,
    label: str,
    value: float,
    top: int,
    color: tuple[int, int, int],
) -> None:
    left = 22
    width = canvas.shape[1] - 44
    cv2.putText(
        canvas,
        label,
        (left, top - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.50,
        (180, 180, 180),
        1,
        cv2.LINE_AA,
    )
    cv2.rectangle(canvas, (left, top), (left + width, top + 22), (70, 70, 70), 1)
    cv2.rectangle(
        canvas,
        (left, top),
        (left + int(width * max(0.0, min(1.0, value))), top + 22),
        color,
        -1,
    )


def _show_camera_window(
    frame: NDArray[np.uint8],
    attention_result,
    attention_state: AttentionState,
    gaze_duration: float,
    emotion: EmotionState | None,
) -> None:
    camera_view = frame.copy()
    draw_attention_overlay(
        camera_view,
        attention_result,
        attention_state,
        gaze_duration,
    )
    draw_emotion_overlay(camera_view, emotion, origin=(20, 110))
    cv2.imshow(CAMERA_WINDOW_NAME, camera_view)


def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Não foi possível abrir a webcam no índice {CAMERA_INDEX}."
        )

    moktak: MoktakSession | None = None
    try:
        effect_profile = build_effect_profile_from_env(
            env_float=env_float,
            env_int=env_int,
        )
        moktak = MoktakSession(effect_profile=effect_profile)
        gaze_tracker = GazeDurationTracker()
        activation_tracker = AttentionActivationTracker()
        positive_tracker = PositiveEmotionHoldTracker(
            effect_profile.positive_hold_seconds,
        )
        positive_meter = PositiveDurationMeter()

        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(EMOTION_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(MOKTAK_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(CAMERA_WINDOW_NAME, 40, 80)
        cv2.moveWindow(EMOTION_WINDOW_NAME, 760, 80)
        cv2.moveWindow(MOKTAK_WINDOW_NAME, 1320, 80)

        moktak.start()
        with (
            AttentionDetector() as attention_detector,
            EmotionDetector() as emotion_detector,
        ):
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Não foi possível capturar um frame da webcam.")
                    break

                now = time.monotonic()
                attention_result = attention_detector.process(frame)
                attention_state = AttentionState.classify(
                    attention_result.score,
                    face_detected=attention_result.face_box is not None,
                )
                gaze_duration = gaze_tracker.update(attention_state, now)
                sustained_attention = (
                    attention_state is AttentionState.ATTENDING
                    and gaze_duration >= ATTENTION_DURATION_SECONDS
                )
                _handle_moktak_attention_event(
                    state=attention_state,
                    sustained_attention=sustained_attention,
                    activation_tracker=activation_tracker,
                    moktak=moktak,
                )

                emotion = emotion_detector.detect(frame)
                positive_seconds = positive_meter.update(emotion, now)
                if moktak.state is MoktakState.STOPPED:
                    positive_tracker.reset()
                    positive_meter.reset()
                    positive_seconds = 0.0

                _handle_moktak_emotion_event(
                    emotion=emotion,
                    timestamp=now,
                    positive_tracker=positive_tracker,
                    moktak=moktak,
                )

                emotion_label = emotion.label if emotion else "none"
                emotion_confidence = emotion.confidence if emotion else 0.0
                print(
                    "moktak="
                    f"{moktak.state.name} attention={attention_state.name} "
                    f"sustained_attention_seconds={gaze_duration:.2f} "
                    f"emotion={emotion_label} "
                    f"emotion_confidence={emotion_confidence:.3f} "
                    f"positive_seconds={positive_seconds:.2f}",
                    flush=True,
                )

                _show_camera_window(
                    frame,
                    attention_result,
                    attention_state,
                    gaze_duration,
                    emotion,
                )
                # cv2.imshow(EMOTION_WINDOW_NAME, render_emotion_debug(emotion))
                cv2.imshow(
                    MOKTAK_WINDOW_NAME,
                    _render_moktak_debug(
                        attention_state=attention_state,
                        gaze_duration=gaze_duration,
                        sustained_attention=sustained_attention,
                        emotion=emotion,
                        positive_seconds=positive_seconds,
                        positive_hold_seconds=effect_profile.positive_hold_seconds,
                        moktak_state=moktak.state,
                    ),
                )

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        if moktak is not None:
            moktak.shutdown()
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
