"""Unit tests for deterministic attention-score calculations."""

import unittest

import cv2
import numpy as np

from src.attention.detector import (
    AttentionComponents,
    _compose_score,
    _head_pose_score,
)


class AttentionScoreTests(unittest.TestCase):
    def test_component_weights_sum_to_one(self) -> None:
        components = AttentionComponents(1.0, 1.0, 1.0, 1.0, 1.0)
        self.assertEqual(_compose_score(components), 1.0)

    def test_head_facing_camera_has_full_pose_score(self) -> None:
        self.assertAlmostEqual(_head_pose_score(np.eye(4)), 1.0)

    def test_large_yaw_is_rejected(self) -> None:
        rotation, _ = cv2.Rodrigues(np.array((0.0, np.deg2rad(35.0), 0.0)))
        transformation = np.eye(4)
        transformation[:3, :3] = rotation
        self.assertLess(_head_pose_score(transformation), 0.2)

    def test_composed_score_is_clamped(self) -> None:
        components = AttentionComponents(2.0, 2.0, 2.0, 2.0, 2.0)
        self.assertEqual(_compose_score(components), 1.0)


if __name__ == "__main__":
    unittest.main()
