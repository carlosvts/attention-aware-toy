"""Webcam application for the social-perception HRI MVP."""

from concurrent.futures import Future, ThreadPoolExecutor
import time

import cv2
import numpy as np
from numpy.typing import NDArray

from src.attention import (
    AttentionDetector,
    AttentionResult,
    AttentionSessionGate,
    AttentionState,
    GazeDurationTracker,
)
from src.llm import ensure_ollama_ready, generate_response, unload_models
from src.profiling import profile_block, profile_interaction
from src.vision import describe_scene

CAMERA_INDEX = 0
ATTENTION_THRESHOLD = 0.7
ATTENTION_DURATION_SECONDS = 1.0
ATTENTION_RELEASE_SECONDS = 1.0
COOLDOWN_SECONDS = 5.0
WINDOW_NAME = "Social Perception HRI"


def _draw_attention_features(
    frame: NDArray[np.uint8], result: AttentionResult
) -> None:
    """Draw the facial evidence used by the attention score."""
    features = result.features
    if features is None:
        return

    eye_color = (80, 220, 80)
    iris_color = (255, 80, 220)
    nose_color = (0, 220, 255)
    for contour in (features.right_eye, features.left_eye):
        cv2.polylines(
            frame,
            [np.asarray(contour, dtype=np.int32)],
            True,
            eye_color,
            1,
            cv2.LINE_AA,
        )
    for center, radius in (features.right_iris, features.left_iris):
        cv2.circle(frame, center, radius, iris_color, 2, cv2.LINE_AA)
        cv2.circle(frame, center, 2, iris_color, -1, cv2.LINE_AA)

    cv2.line(
        frame,
        features.nose_reference[0],
        features.nose_reference[1],
        nose_color,
        1,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        frame,
        features.nose,
        nose_color,
        cv2.MARKER_CROSS,
        12,
        2,
        cv2.LINE_AA,
    )

    origin, x_axis, y_axis, z_axis = features.head_axes
    for endpoint, axis_color, name in (
        (x_axis, (0, 0, 255), "X"),
        (y_axis, (0, 255, 0), "Y"),
        (z_axis, (255, 0, 0), "Z"),
    ):
        cv2.arrowedLine(
            frame, origin, endpoint, axis_color, 2, cv2.LINE_AA, tipLength=0.18
        )
        cv2.putText(
            frame,
            name,
            endpoint,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            axis_color,
            1,
            cv2.LINE_AA,
        )


def generate_interaction(
    frame: NDArray[np.uint8],
    attention_state: AttentionState,
    gaze_duration: float,
) -> tuple[str, str]:
    """Describe a frame and generate the robot's textual response."""
    with profile_interaction():
        scene_description = describe_scene(frame)
        response = generate_response(
            scene_description, attention_state, gaze_duration
        )
        return scene_description, response


def draw_overlay(
    frame: NDArray[np.uint8],
    result: AttentionResult,
    attention_state: AttentionState,
    gaze_duration: float,
) -> None:
    """Draw face location and attention state on the preview frame."""
    state_colors = {
        AttentionState.NO_FACE: (120, 120, 120),
        AttentionState.FACE_DETECTED: (0, 180, 255),
        AttentionState.LOOKING_BRIEFLY: (0, 220, 220),
        AttentionState.DISTRACTED: (0, 140, 255),
        AttentionState.ATTENDING: (80, 220, 80),
    }
    color = state_colors[attention_state]
    if result.face_box is not None:
        x, y, width, height = result.face_box
        cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
    _draw_attention_features(frame, result)

    label = f"state: {attention_state.name}  score: {result.score:.2f}"
    cv2.putText(frame, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    if result.components is not None:
        detail = (
            f"head: {result.components.head_pose:.2f}  "
            f"gaze: {result.components.iris_direction:.2f}"
        )
        cv2.putText(
            frame, detail, (20, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2
        )
    if result.features is not None:
        pitch, yaw, roll = result.features.head_angles
        pose = f"pitch: {pitch:+.0f}  yaw: {yaw:+.0f}  roll: {roll:+.0f} deg"
        cv2.putText(
            frame, pose, (20, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2
        )
    cv2.putText(
        frame,
        f"gaze duration: {gaze_duration:.1f}s",
        (20, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
    )


def run() -> None:
    """Run the webcam attention and response loop."""
    ensure_ollama_ready()
    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(
            f"Não foi possível abrir a webcam no índice {CAMERA_INDEX}."
        )

    gaze_tracker = GazeDurationTracker()
    session_gate = AttentionSessionGate(
        threshold=ATTENTION_THRESHOLD,
        release_duration=ATTENTION_RELEASE_SECONDS,
    )
    last_response_at = float("-inf")
    response_worker = ThreadPoolExecutor(max_workers=1)
    pending_response: Future[tuple[str, str]] | None = None

    try:
        with AttentionDetector() as detector:
            while True:
                with profile_block("frame_capture"):
                    ok, frame = camera.read()
                if not ok:
                    print("Não foi possível capturar um frame da webcam.")
                    break

                now = time.monotonic()
                result = detector.process(frame)
                attention_state = AttentionState.classify(
                    result.score, face_detected=result.face_box is not None
                )
                gaze_duration = gaze_tracker.update(attention_state, now)
                sustained_attention = (
                    attention_state is AttentionState.ATTENDING
                    and gaze_duration >= ATTENTION_DURATION_SECONDS
                )
                new_attention_session = session_gate.update(
                    score=result.score,
                    timestamp=now,
                    sustained_attention=sustained_attention,
                )

                if pending_response is not None and pending_response.done():
                    try:
                        scene_description, response = pending_response.result()
                        print(f"Cena: {scene_description}")
                        print(f"Resposta: {response}")
                    except Exception as error:
                        print(f"Erro inesperado ao gerar resposta: {error}")
                    pending_response = None

                cooldown_finished = now - last_response_at >= COOLDOWN_SECONDS
                if (
                    new_attention_session
                    and cooldown_finished
                    and pending_response is None
                ):
                    captured_frame = frame.copy()
                    print("Processando cena...")
                    pending_response = response_worker.submit(
                        generate_interaction,
                        captured_frame,
                        attention_state,
                        gaze_duration,
                    )
                    last_response_at = now

                draw_overlay(frame, result, attention_state, gaze_duration)
                cv2.imshow(WINDOW_NAME, frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        camera.release()
        cv2.destroyAllWindows()
        # A running request cannot be cancelled. Wait for it before unloading,
        # otherwise it could reload the model after the cleanup request.
        response_worker.shutdown(wait=True, cancel_futures=True)
        unload_models()


if __name__ == "__main__":
    run()
