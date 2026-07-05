"""Apparent facial-expression detection with MediaPipe blendshapes."""

from pathlib import Path
import time

import cv2
import mediapipe as mp
from numpy.typing import NDArray

from .types import EmotionState


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

# averages the score of a specific blendshape 
def _score(blendshapes: dict[str, float], *names: str) -> float:
    if not names:
        return 0.0
    return sum(blendshapes.get(name, 0.0) for name in names) / len(names)


def classify_expression(blendshapes: dict[str, float]) -> EmotionState:
    """Map MediaPipe blendshapes to cautious apparent-expression labels."""
    smile = _score(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _score(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    eye_squint = _score(blendshapes, "eyeSquintLeft", "eyeSquintRight")

    brow_down = _score(blendshapes, "browDownLeft", "browDownRight")
    brow_outer_up = _score(blendshapes, "browOuterUpRight", "browOuterUpLeft")
    eye_wide = _score(blendshapes, "eyeWideLeft", "eyeWideRight")
    jaw_open = _score(blendshapes, "jawOpen")
    surprised = (
        0.45 * brow_outer_up
        + 0.40 * eye_wide
        + 0.15 * jaw_open
    )
    positive = max(
        smile if smile >= 0.30 else 0.0,
        surprised if surprised >= 0.22 else 0.0,
    )
    shrug = _score(blendshapes, "mouthShrugUpper", "mouthShrugLower")
    mouth_shrug_lower = _score(blendshapes, "mouthShrugLower")
    mouth_pucker = _score(blendshapes, "mouthPucker")
    mouth_press = _score(blendshapes, "mouthPressLeft", "mouthPressRight")
    nose_sneer = _score(
        blendshapes,
        "noseSneerLeft",
        "noseSneerRight",
    )
    upper_lip = _score(
        blendshapes,
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
    )
    # A negative expression must have mutually supporting cues. The activation
    # floor rejects isolated blendshapes, while the weighted averages retain
    # information from every active cue instead of collapsing to the weakest.
    activation_floor = 0.08

    mouth_tension = max(mouth_press, frown)
    tension_support = 0.70 * brow_down + 0.30 * eye_squint
    tension = 0.0
    if (
        mouth_tension >= activation_floor
        and max(brow_down, eye_squint) >= activation_floor
    ):
        tension = 0.55 * mouth_tension + 0.45 * tension_support

    negative_mouth_support = max(
        upper_lip,
        mouth_press,
        frown,
        mouth_shrug_lower,
    )
    nasal_negative = 0.0
    if (
        nose_sneer >= activation_floor
        and negative_mouth_support >= activation_floor
    ):
        nasal_negative = 0.60 * nose_sneer + 0.40 * negative_mouth_support

    sadness_cues = (
        (mouth_shrug_lower, 0.60),
        (frown, 0.40),
    )
    sadness = 0.0
    active_sadness_cues = [
        (cue, weight)
        for cue, weight in sadness_cues
        if cue >= activation_floor
    ]
    if len(active_sadness_cues) >= 2:
        active_weight = sum(weight for _, weight in active_sadness_cues)
        sadness_base = sum(
            cue * weight for cue, weight in active_sadness_cues
        ) / active_weight
        sadness = (
            0.90 * sadness_base
            + 0.10 * mouth_pucker
        )

    negative = max(tension, nasal_negative, sadness)

    candidates = {
        "positive_expression": positive,
        "negative_expression": negative,
    }
    print(
        f"smile={smile:.3f} surprised={surprised:.3f} positive={positive:.3f} "
        f"frown={frown:.3f} brow_down={brow_down:.3f} "
        f"mouth_press={mouth_press:.3f} shrug={shrug:.3f} "
        f"nose_sneer={nose_sneer:.3f} negative={negative:.3f}"
    )

    thresholds = {
        "positive_expression": 0.22,
        "negative_expression": 0.16,
    }

    label, confidence = max(candidates.items(), key=lambda item: item[1])
    if confidence < thresholds[label]:
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

        # 0 is the first face
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
