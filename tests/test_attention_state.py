"""Tests for attention-state classification and gaze timing."""

import unittest

from src.attention import AttentionState, GazeDurationTracker


class AttentionStateTests(unittest.TestCase):
    def test_enum_values_are_ordered(self) -> None:
        self.assertEqual(
            [state.value for state in AttentionState],
            [0, 1, 2, 3, 4],
        )

    def test_score_ranges(self) -> None:
        cases = (
            (0.0, False, AttentionState.NO_FACE),
            (0.0, True, AttentionState.FACE_DETECTED),
            (0.3, True, AttentionState.FACE_DETECTED),
            (0.4, True, AttentionState.LOOKING_BRIEFLY),
            (0.6, True, AttentionState.DISTRACTED),
            (0.8, True, AttentionState.ATTENDING),
        )
        for score, face_detected, expected in cases:
            with self.subTest(score=score, face_detected=face_detected):
                self.assertIs(
                    AttentionState.classify(score, face_detected), expected
                )

    def test_gaze_duration_resets_outside_attending(self) -> None:
        tracker = GazeDurationTracker()
        self.assertEqual(tracker.update(AttentionState.ATTENDING, 10.0), 0.0)
        self.assertAlmostEqual(
            tracker.update(AttentionState.ATTENDING, 11.2), 1.2
        )
        self.assertEqual(tracker.update(AttentionState.DISTRACTED, 11.3), 0.0)
        self.assertEqual(tracker.update(AttentionState.ATTENDING, 12.0), 0.0)


if __name__ == "__main__":
    unittest.main()
