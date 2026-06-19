"""Landmark-based visual-attention estimation using MediaPipe."""

from dataclasses import dataclass
from math import hypot
from pathlib import Path
import time
from typing import Sequence

import cv2
import mediapipe as mp
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AttentionComponents:
    """Normalized signals used to compose an attention score."""

    head_pose: float
    iris_direction: float
    nose_position: float
    eye_symmetry: float
    face_position: float


@dataclass(frozen=True)
class AttentionFeatures:
    """Pixel-space geometry used to explain the attention estimate."""

    nose: tuple[int, int]
    nose_reference: tuple[tuple[int, int], tuple[int, int]]
    right_eye: tuple[tuple[int, int], ...]
    left_eye: tuple[tuple[int, int], ...]
    right_iris: tuple[tuple[int, int], int]
    left_iris: tuple[tuple[int, int], int]
    head_axes: tuple[tuple[int, int], ...]
    head_angles: tuple[float, float, float]


@dataclass(frozen=True)
class AttentionResult:
    """Attention estimate for one video frame."""

    score: float
    face_box: tuple[int, int, int, int] | None
    components: AttentionComponents | None = None
    features: AttentionFeatures | None = None


_WEIGHTS = AttentionComponents(
    head_pose=0.40,
    iris_direction=0.30,
    nose_position=0.15,
    eye_symmetry=0.10,
    face_position=0.05,
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _point(landmarks: Sequence[object], index: int) -> NDArray[np.float64]:
    landmark = landmarks[index]
    return np.array((landmark.x, landmark.y), dtype=np.float64)


def _distance(landmarks: Sequence[object], first: int, second: int) -> float:
    return float(np.linalg.norm(_point(landmarks, first) - _point(landmarks, second)))


def _head_rotation(
    matrix: NDArray[np.float64],
) -> tuple[NDArray[np.float64], tuple[float, float, float]]:
    """Return an orthogonal rotation and signed pitch/yaw/roll angles."""
    rotation_with_scale = np.asarray(matrix, dtype=np.float64)[:3, :3]
    # Remove the small scale/shear component before extracting Euler angles.
    u, _, vh = np.linalg.svd(rotation_with_scale)
    rotation = u @ vh
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vh
    angles = tuple(float(value) for value in cv2.RQDecomp3x3(rotation)[0])
    return rotation, angles


def _head_pose_score(matrix: NDArray[np.float64]) -> float:
    """Score a MediaPipe facial transformation (neutral pose is identity)."""
    _, angles = _head_rotation(matrix)
    pitch, yaw, roll = (abs(value) for value in angles)

    # Looking sideways/up/down is stronger evidence than a slight head roll.
    pitch_score = _clamp01(1.0 - pitch / 22.0)
    yaw_score = _clamp01(1.0 - yaw / 28.0)
    roll_score = _clamp01(1.0 - roll / 30.0)
    weighted_mean = 0.40 * pitch_score + 0.45 * yaw_score + 0.15 * roll_score
    # A single large rotation must not be hidden by the other two neutral axes.
    return 0.70 * min(pitch_score, yaw_score, roll_score) + 0.30 * weighted_mean


def _iris_in_eye_score(
    landmarks: Sequence[object],
    iris_indices: tuple[int, ...],
    corner_indices: tuple[int, int],
    lid_indices: tuple[int, int],
) -> float:
    iris = np.mean([_point(landmarks, index) for index in iris_indices], axis=0)
    corner_a, corner_b = (_point(landmarks, index) for index in corner_indices)
    top, bottom = (_point(landmarks, index) for index in lid_indices)

    eye_axis = corner_b - corner_a
    lid_axis = bottom - top
    horizontal = float(np.dot(iris - corner_a, eye_axis) / max(np.dot(eye_axis, eye_axis), 1e-9))
    vertical = float(np.dot(iris - top, lid_axis) / max(np.dot(lid_axis, lid_axis), 1e-9))
    horizontal_score = _clamp01(1.0 - abs(horizontal - 0.5) / 0.23)
    vertical_score = _clamp01(1.0 - abs(vertical - 0.5) / 0.35)
    return 0.75 * horizontal_score + 0.25 * vertical_score


def _iris_direction_score(landmarks: Sequence[object]) -> float:
    if len(landmarks) < 478:
        return 0.0
    right = _iris_in_eye_score(landmarks, (468, 469, 470, 471, 472), (33, 133), (159, 145))
    left = _iris_in_eye_score(landmarks, (473, 474, 475, 476, 477), (362, 263), (386, 374))
    right_opening = _distance(landmarks, 159, 145) / max(
        _distance(landmarks, 33, 133), 1e-9
    )
    left_opening = _distance(landmarks, 386, 374) / max(
        _distance(landmarks, 362, 263), 1e-9
    )
    visibility = _clamp01(min(right_opening, left_opening) / 0.12)
    return visibility * (right + left) / 2.0


def _nose_position_score(landmarks: Sequence[object], face_width: float) -> float:
    nose = _point(landmarks, 1)
    eye_midpoint = np.mean(
        [_point(landmarks, index) for index in (33, 133, 362, 263)], axis=0
    )
    chin = _point(landmarks, 152)

    horizontal_score = _clamp01(1.0 - abs(nose[0] - eye_midpoint[0]) / max(0.20 * face_width, 1e-9))
    eye_to_chin = chin - eye_midpoint
    vertical_ratio = float(
        np.dot(nose - eye_midpoint, eye_to_chin)
        / max(np.dot(eye_to_chin, eye_to_chin), 1e-9)
    )
    vertical_score = _clamp01(1.0 - abs(vertical_ratio - 0.43) / 0.24)
    return 0.70 * horizontal_score + 0.30 * vertical_score


def _eye_symmetry_score(landmarks: Sequence[object]) -> float:
    right_width = _distance(landmarks, 33, 133)
    left_width = _distance(landmarks, 362, 263)
    right_opening = _distance(landmarks, 159, 145) / max(right_width, 1e-9)
    left_opening = _distance(landmarks, 386, 374) / max(left_width, 1e-9)

    width_similarity = min(right_width, left_width) / max(right_width, left_width, 1e-9)
    opening_similarity = min(right_opening, left_opening) / max(
        right_opening, left_opening, 1e-9
    )
    # Closed or almost closed eyes should not look attentive even if symmetric.
    opening_score = _clamp01(min(right_opening, left_opening) / 0.16)
    return opening_score * (0.70 * opening_similarity + 0.30 * width_similarity)


def _face_position_score(center_x: float, center_y: float) -> float:
    normalized_distance = hypot(center_x - 0.5, center_y - 0.5) / hypot(0.5, 0.5)
    return _clamp01(1.0 - normalized_distance)


def _compose_score(components: AttentionComponents) -> float:
    return _clamp01(sum(
        getattr(components, field) * getattr(_WEIGHTS, field)
        for field in AttentionComponents.__dataclass_fields__
    ))


def _attention_features(
    landmarks: Sequence[object],
    transformation: NDArray[np.float64],
    frame_width: int,
    frame_height: int,
    face_width: float,
) -> AttentionFeatures:
    """Convert the landmarks involved in scoring into drawable geometry."""
    def pixel(point: NDArray[np.float64]) -> tuple[int, int]:
        return (
            int(np.clip(point[0] * frame_width, 0, frame_width - 1)),
            int(np.clip(point[1] * frame_height, 0, frame_height - 1)),
        )

    def pixels(indices: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        return tuple(pixel(_point(landmarks, index)) for index in indices)

    def iris(indices: tuple[int, ...]) -> tuple[tuple[int, int], int]:
        points = [_point(landmarks, index) for index in indices]
        center = np.mean(points, axis=0)
        radii = [
            hypot(
                (point[0] - center[0]) * frame_width,
                (point[1] - center[1]) * frame_height,
            )
            for point in points[1:]
        ]
        return pixel(center), max(2, int(round(float(np.mean(radii)))))

    nose_point = _point(landmarks, 1)
    eye_midpoint = np.mean(
        [_point(landmarks, index) for index in (33, 133, 362, 263)], axis=0
    )
    rotation, angles = _head_rotation(transformation)
    origin = np.array(pixel(nose_point), dtype=np.float64)
    axis_scale = max(20.0, face_width * frame_width * 0.28)
    axis_ends = tuple(
        tuple(np.rint(origin + rotation[:2, axis] * axis_scale).astype(int))
        for axis in range(3)
    )

    return AttentionFeatures(
        nose=tuple(origin.astype(int)),
        nose_reference=(pixel(eye_midpoint), pixel(_point(landmarks, 152))),
        right_eye=pixels((33, 160, 158, 133, 153, 144)),
        left_eye=pixels((362, 385, 387, 263, 373, 380)),
        right_iris=iris((468, 469, 470, 471, 472)),
        left_iris=iris((473, 474, 475, 476, 477)),
        head_axes=(tuple(origin.astype(int)), *axis_ends),
        head_angles=angles,
    )


class AttentionDetector:
    """Estimate whether the largest visible face is looking near the camera."""

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        model_path: Path | None = None,
    ) -> None:
        if model_path is None:
            model_path = (
                Path(__file__).resolve().parents[2]
                / "models"
                / "face_landmarker.task"
            )
        if not model_path.is_file():
            raise FileNotFoundError(f"Modelo MediaPipe não encontrado: {model_path}")

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=4,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_detection_confidence,
            output_facial_transformation_matrixes=True,
        )
        self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(
            options
        )
        self._last_timestamp_ms = -1

    def process(self, frame: NDArray[np.uint8]) -> AttentionResult:
        """Return a weighted gaze/pose score and the largest detected face box."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        media_pipe_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = max(self._last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
        self._last_timestamp_ms = timestamp_ms
        result = self._face_landmarker.detect_for_video(media_pipe_image, timestamp_ms)
        if not result.face_landmarks:
            return AttentionResult(score=0.0, face_box=None)

        bounds = []
        for landmarks in result.face_landmarks:
            xs = [landmark.x for landmark in landmarks]
            ys = [landmark.y for landmark in landmarks]
            bounds.append((min(xs), min(ys), max(xs), max(ys)))
        index = max(
            range(len(bounds)),
            key=lambda item: (bounds[item][2] - bounds[item][0])
            * (bounds[item][3] - bounds[item][1]),
        )
        landmarks = result.face_landmarks[index]
        min_x, min_y, max_x, max_y = bounds[index]
        face_width = max_x - min_x
        components = AttentionComponents(
            head_pose=_head_pose_score(result.facial_transformation_matrixes[index]),
            iris_direction=_iris_direction_score(landmarks),
            nose_position=_nose_position_score(landmarks, face_width),
            eye_symmetry=_eye_symmetry_score(landmarks),
            face_position=_face_position_score((min_x + max_x) / 2, (min_y + max_y) / 2),
        )

        frame_height, frame_width = frame.shape[:2]
        features = _attention_features(
            landmarks,
            result.facial_transformation_matrixes[index],
            frame_width,
            frame_height,
            face_width,
        )
        x1 = max(0, min(frame_width - 1, int(min_x * frame_width)))
        y1 = max(0, min(frame_height - 1, int(min_y * frame_height)))
        x2 = max(x1, min(frame_width, int(max_x * frame_width)))
        y2 = max(y1, min(frame_height, int(max_y * frame_height)))
        return AttentionResult(
            score=_compose_score(components),
            face_box=(x1, y1, x2 - x1, y2 - y1),
            components=components,
            features=features,
        )

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._face_landmarker.close()

    def __enter__(self) -> "AttentionDetector":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
