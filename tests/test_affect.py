"""Tests for heuristic apparent-affect mapping."""

import unittest

from src.emotions import classify_expression


class AffectHeuristicTests(unittest.TestCase):
    def test_smile_maps_to_smiling_apparent_expression(self) -> None:
        state = classify_expression({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7})

        self.assertEqual(state.label, "smiling_expression")
        self.assertGreater(state.confidence, 0.7)

    def test_brow_down_maps_to_focused_apparent_expression(self) -> None:
        state = classify_expression({"browDownLeft": 0.65, "browDownRight": 0.75})

        self.assertEqual(state.label, "focused_expression")

    def test_raised_brows_eye_wide_and_jaw_open_map_to_surprised_expression(self) -> None:
        state = classify_expression(
            {
                "browInnerUp": 0.9,
                "browOuterUpRight": 0.8,
                "browOuterUpLeft": 0.7,
                "eyeWideLeft": 0.65,
                "eyeWideRight": 0.75,
                "jawOpen": 0.55,
            }
        )

        self.assertEqual(state.label, "surprised_expression")
        self.assertGreater(state.confidence, 0.6)

    def test_low_scores_map_to_neutral_expression(self) -> None:
        state = classify_expression({})

        self.assertEqual(state.label, "neutral_expression")
        self.assertGreaterEqual(state.confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
