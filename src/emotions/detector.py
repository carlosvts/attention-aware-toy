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
    """
        Facial expression heuristic based on FACS Action Units (AUs), approximated
        using MediaPipe Face Landmarker blendshapes.

        Positive (happiness):
        - AU12 (Lip Corner Puller) -> mouthSmileLeft/Right
        - AU6 (Cheek Raiser) -> cheekSquintLeft/Right

        Sadness:
        - AU1 (Inner Brow Raiser) -> browInnerUp
        - AU4 (Brow Lowerer) -> browDownLeft/Right
        - AU15 (Lip Corner Depressor) -> mouthFrownLeft/Right

        Anger / tension:
        - AU4 (Brow Lowerer) -> browDownLeft/Right
        - AU7 (Lid Tightener) -> eyeSquintLeft/Right
        - AU23/24 (Lip Tightener/Pressor) -> mouthPressLeft/Right, mouthClose

        Disgust / aversion:
        - AU9 (Nose Wrinkler) -> noseSneerLeft/Right
        - AU10 (Upper Lip Raiser) -> mouthUpperUpLeft/Right
        - AU15 (Lip Corner Depressor) -> mouthFrownLeft/Right

        Negative expressions require both upper- and lower-face activation to reduce
        false positives from isolated blendshape activations.

        Sources:
        - Ekman, Friesen & Hager, Facial Action Coding System (FACS), 2002.
        - MediaPipe Face Landmarker documentation.
        - iMotions FACS guide.
        - Noldus FaceReader/FACS reference.
    """
    smile = _score(blendshapes, "mouthSmileLeft", "mouthSmileRight")
    frown = _score(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    eye_squint = _score(blendshapes, "eyeSquintLeft", "eyeSquintRight")
    mouth_close = _score(blendshapes, "mouthClose")
    mouth_frown = _score(blendshapes, "mouthFrownLeft", "mouthFrownRight")
    brow_inner_up = _score(blendshapes, "browInnerUp")
    brow_down = _score(blendshapes, "browDownLeft", "browDownRight")
    brow_outer_up = _score(blendshapes, "browOuterUpRight", "browOuterUpLeft")
    eye_wide = _score(blendshapes, "eyeWideLeft", "eyeWideRight")
    jaw_open = _score(blendshapes, "jawOpen")
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

    upper_negative = max(brow_down, eye_squint, brow_inner_up)
    lower_negative = max(
        mouth_frown,
        mouth_press,
        nose_sneer,
        upper_lip,
    )

    """
        Since negative expressions usually have more than one muscle in the face active
        this variables will ensure that a negative expression can only be
        evaluated if we have, e.g a brow AND mouth signal indicating that. Avoiding false positives
    """
    # has_cross_face_support = (
    #    upper_negative >= 0.08
    #    and lower_negative >= 0.08
    #)
    has_sadness_support = (
        mouth_frown >= 0.025
        and max(brow_inner_up, brow_down) >= 0.08
    )

    has_disgust_support = (
        nose_sneer >= 0.06
        and max(upper_lip, mouth_frown) >= 0.025
    )

    has_tension_support = (
        brow_down >= 0.18
        and eye_squint >= 0.16
        and max(mouth_press, mouth_close, mouth_frown, mouth_shrug_lower) >= 0.005
    )

    anger_tension = 0.0
    if brow_down >= 0.10 and max(eye_squint, mouth_press) >= 0.08:
        anger_tension = (
            0.45 * brow_down
            + 0.20 * eye_squint
            + 0.25 * mouth_press
            + 0.10 * mouth_close
        )

    sadness = 0.0
    if mouth_frown >= 0.10 and max(brow_inner_up, brow_down) >= 0.08:
        sadness = (
            0.50 * mouth_frown
            + 0.30 * brow_inner_up
            + 0.15 * brow_down
            + 0.05 * mouth_shrug_lower
        )

    disgust_aversion = 0.0
    if nose_sneer >= 0.10 and max(upper_lip, mouth_frown) >= 0.08:
        disgust_aversion = (
            0.50 * nose_sneer
            + 0.30 * upper_lip
            + 0.20 * mouth_frown
        )

    anxiety_tension = 0.0
    if mouth_press >= 0.10 and max(brow_down, eye_squint) >= 0.08:
        anxiety_tension = (
            0.40 * mouth_press
            + 0.25 * eye_squint
            + 0.25 * brow_down
            + 0.10 * mouth_close
        )

    # This is the best way i get to evaluate a negative expression, idk if its optimal tho
    # TODO: Maybe add some metric that adds up primary_emotion + sum_of_others
    # e.g Primary is the greater one (example: sadness), so negative will be sadness * 0.80 + sum(others) * 0.20
    raw_negative = max(
        anger_tension     if has_tension_support else 0.0,
        sadness           if has_sadness_support else 0.0,
        disgust_aversion  if has_disgust_support else 0.0,
        anxiety_tension   if has_tension_support else 0.0,
    )

    surprised = (
        0.45 * brow_outer_up
        + 0.40 * eye_wide
        + 0.15 * jaw_open
    )

    positive_interference = max(smile, surprised)

    if positive_interference >= 0.25:
        raw_negative *= 0.65

    if positive_interference >= 0.35:
        raw_negative *= 0.45

    negative = raw_negative
    positive = max(
        smile if smile >= 0.28 else 0.0,
        surprised if surprised >= 0.24 else 0.0,
    )

    candidates = {
        "positive_expression": positive,
        "negative_expression": negative,
    }

    # TODO: REMINDER: Maybe useful later?
    negative_type, negative_type_score = max(
        {
            "anger_tension": anger_tension,
            "sadness": sadness,
            "disgust_aversion": disgust_aversion,
            "anxiety_tension": anxiety_tension,
        }.items(),
        key=lambda item: item[1],
    )

    # DEBUG
    print(
        f"positive={positive:.3f} "
        f"negative={negative:.3f} "
        f"negative_type={negative_type}:{negative_type_score:.3f} "
        f"anger_tension={anger_tension:.3f} "
        f"sadness={sadness:.3f} "
        f"disgust_aversion={disgust_aversion:.3f} "
        f"anxiety_tension={anxiety_tension:.3f} "
        f"brow_down={brow_down:.3f} "
        f"brow_inner_up={brow_inner_up:.3f} "
        f"eye_squint={eye_squint:.3f} "
        f"mouth_frown={mouth_frown:.3f} "
        f"mouth_press={mouth_press:.3f} "
        f"nose_sneer={nose_sneer:.3f}"
    )
    # END DEBUG

    thresholds = {
        "positive_expression": 0.24,
        "negative_expression": 0.18,
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
