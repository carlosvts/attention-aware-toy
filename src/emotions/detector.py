"""Apparent facial-expression detection with MediaPipe blendshapes."""

from pathlib import Path
import time

import cv2
import mediapipe as mp
from numpy.typing import NDArray

from .types import EmotionState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _score(blendshapes: dict[str, float], *names: str) -> float:
    if not names:
        return 0.0
    return sum(blendshapes.get(name, 0.0) for name in names) / len(names)


def classify_expression(blendshapes: dict[str, float]) -> EmotionState:
    """Map MediaPipe blendshapes to cautious apparent-expression labels."""
    smile = _score(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _score(blendshapes, "mouthFrownLeft", "mouthFrownRight")

    # For surprising
    brow_down = _score(blendshapes, "browDownLeft", "browDownRight")
    brow_up = _score(
        blendshapes,
        "browInnerUp",
        "browOuterUpRight",
        "browOuterUpLeft",
    )
    eye_wide = _score(blendshapes, "eyeWideLeft", "eyeWideRight")
    jaw_open = _score(blendshapes, "jawOpen")
    surprised = (
        0.45 * brow_up
        + 0.35 * eye_wide
        + 0.20 * jaw_open
    )

    candidates = {
        "smiling_expression": smile,
        "frowning_expression": frown,
        "focused_expression": brow_down,
        "surprised_expression": surprised,
    }
    label, confidence = max(candidates.items(), key=lambda item: item[1])
    if confidence < 0.28:
        label = "neutral_expression"
        confidence = max(0.45, 1.0 - max(candidates.values()))

    return EmotionState(
        label=label,
        confidence=_clamp01(confidence),
        blendshapes=blendshapes,
    )


class EmotionDetector:
    """Detect apparent facial expression from one OpenCV BGR frame."""

    def __init__(self, model_path: str = "models/face_landmarker.task") -> None:
        path = Path(model_path)
        if not path.is_file():
            path = Path(__file__).resolve().parents[2] / model_path
        if not path.is_file():
            raise FileNotFoundError(f"Modelo MediaPipe não encontrado: {path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=True,
        )
        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self._last_timestamp_ms = -1

    def detect(self, frame_bgr: NDArray) -> EmotionState | None:
        """Return apparent expression for a BGR frame, or None when no face exists."""
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms

        result = self._face_landmarker.detect_for_video(image, timestamp_ms)
        if not result.face_landmarks or not result.face_blendshapes:
            return None

        blendshapes = {
            category.category_name: float(category.score)
            for category in result.face_blendshapes[0]
        }
        return classify_expression(blendshapes)

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_landmarker.close()

    def __enter__(self) -> "EmotionDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
