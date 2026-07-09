"""Manual webcam test for emotion/attention-driven moktak audio.

Run from the project root with: python -m tests.test_moktak
"""

from pathlib import Path
import sys
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention import AttentionDetector, AttentionState, GazeDurationTracker
from src.audio import (
    MoktakParameters,
    MoktakPlayer,
    decide_moktak,
    render_moktak_visualizer,
)
from src.debug import draw_attention_overlay, draw_emotion_overlay
from src.emotions import EmotionDetector


CAMERA_INDEX = 0
CAMERA_WINDOW_NAME = "Moktak Camera Test"
AUDIO_WINDOW_NAME = "Moktak Audio Visualizer"
DECISION_STABILITY_SECONDS = 0.45
MAX_SWITCH_WAIT_SECONDS = 1.50
QUIET_LEVEL = 0.015


def main() -> None:
    """Run real detectors and expose their current moktak decision."""
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Não foi possível abrir a webcam no índice {CAMERA_INDEX}."
        )

    cv2.namedWindow(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.namedWindow(AUDIO_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.moveWindow(CAMERA_WINDOW_NAME, 40, 80)
    cv2.moveWindow(AUDIO_WINDOW_NAME, 760, 80)

    gaze_tracker = GazeDurationTracker()
    active_parameters = MoktakParameters.disabled()
    candidate_parameters = active_parameters
    candidate_since = time.monotonic()
    print("Moktak test: continuous audio loop; Q or ESC exits.")

    try:
        with (
            AttentionDetector() as attention_detector,
            EmotionDetector() as emotion_detector,
            MoktakPlayer() as player,
        ):
            while True:
                ok, frame = camera.read()
                if not ok:
                    print("Não foi possível capturar um frame da webcam.")
                    break

                attention_result = attention_detector.process(frame)
                attention_state = AttentionState.classify(
                    attention_result.score,
                    face_detected=attention_result.face_box is not None,
                )
                gaze_duration = gaze_tracker.update(
                    attention_state,
                    time.monotonic(),
                )
                emotion = emotion_detector.detect(frame)

                # This integration test intentionally excludes mocked mudra/LLM.
                detected_parameters = decide_moktak(
                    emotion_state=emotion,
                    attention_state=attention_state,
                )
                now = time.monotonic()
                if not detected_parameters.enabled:
                    if active_parameters.enabled:
                        player.play(detected_parameters)
                    active_parameters = detected_parameters
                    candidate_parameters = detected_parameters
                    candidate_since = now
                else:
                    if detected_parameters != candidate_parameters:
                        candidate_parameters = detected_parameters
                        candidate_since = now

                    candidate_age = now - candidate_since
                    safe_to_switch = (
                        not active_parameters.enabled
                        or player.current_level() <= QUIET_LEVEL
                        or candidate_age >= MAX_SWITCH_WAIT_SECONDS
                    )
                    if (
                        candidate_parameters != active_parameters
                        and candidate_age >= DECISION_STABILITY_SECONDS
                        and safe_to_switch
                    ):
                        player.play(candidate_parameters)
                        active_parameters = candidate_parameters

                camera_view = frame.copy()
                draw_attention_overlay(
                    camera_view,
                    attention_result,
                    attention_state,
                    gaze_duration,
                )
                draw_emotion_overlay(camera_view, emotion, origin=(20, 105))
                cv2.imshow(CAMERA_WINDOW_NAME, camera_view)

                emotion_label = emotion.label if emotion else "none"
                status = (
                    f"emotion={emotion_label} attention={attention_state.name} "
                    "| Q/ESC: exit"
                )
                if candidate_parameters != active_parameters:
                    status = f"{status} | pending audio change"
                if player.last_error:
                    status = f"audio error: {player.last_error}"
                audio_view = render_moktak_visualizer(
                    parameters=active_parameters,
                    waveform=player.waveform(),
                    current_level=player.current_level(),
                    status=status,
                )
                cv2.imshow(AUDIO_WINDOW_NAME, audio_view)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
