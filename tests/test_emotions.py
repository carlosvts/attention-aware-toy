"""Manual webcam debug script for apparent facial-expression detection."""

from pathlib import Path
import sys

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.debug import draw_emotion_overlay, render_emotion_debug
from src.emotions import EmotionDetector

CAMERA_INDEX = 0
CAMERA_WINDOW_NAME = "Emotion Test"
DEBUG_WINDOW_NAME = "Emotion Debug"


def show_camera_window(frame, emotion) -> None:
    """Show the live webcam frame."""
    camera_view = frame.copy()
    draw_emotion_overlay(camera_view, emotion)
    cv2.imshow(CAMERA_WINDOW_NAME, camera_view)


def show_debug_window(emotion) -> None:
    """Show the separate emotion debug panel."""
    debug_view = render_emotion_debug(emotion)
    cv2.imshow(DEBUG_WINDOW_NAME, debug_view)


def main() -> None:
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Não foi possível abrir a webcam no índice {CAMERA_INDEX}."
        )

    try:
        # Create two windows and try to not have an overlay between them
        cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.namedWindow(DEBUG_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(CAMERA_WINDOW_NAME, 40, 80)
        cv2.moveWindow(DEBUG_WINDOW_NAME, 760, 80)

        with EmotionDetector() as detector:
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Não foi possível capturar um frame da webcam.")
                    break

                emotion = detector.detect(frame)
                if emotion is None:
                    print("apparent_expression=none")
                else:
                    print(
                        "apparent_expression="
                        f"{emotion.label} confidence={emotion.confidence:.3f}"
                    )

                show_camera_window(frame, emotion)
                show_debug_window(emotion)

                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
