"""Tests for heuristic apparent-affect mapping."""

import unittest

from src.emotions import classify_expression


class AffectHeuristicTests(unittest.TestCase):
    def test_smile_maps_to_positive_apparent_expression(self) -> None:
        state = classify_expression({"mouthSmileLeft": 0.8, "mouthSmileRight": 0.7})

        self.assertEqual(state.label, "positive_expression")
        self.assertGreater(state.confidence, 0.7)

    def test_brow_down_alone_maps_to_neutral_expression(self) -> None:
        state = classify_expression({"browDownLeft": 0.65, "browDownRight": 0.75})

        self.assertEqual(state.label, "neutral_expression")

    def test_single_negative_cue_maps_to_neutral_expression(self) -> None:
        state = classify_expression(
            {"mouthFrownLeft": 0.8, "mouthFrownRight": 0.8}
        )

        self.assertEqual(state.label, "neutral_expression")

    def test_weak_negative_cues_map_to_neutral_expression(self) -> None:
        state = classify_expression(
            {
                "mouthPressLeft": 0.14,
                "mouthPressRight": 0.14,
                "browDownLeft": 0.14,
                "browDownRight": 0.14,
            }
        )

        self.assertEqual(state.label, "neutral_expression")

    def test_coherent_negative_cues_map_to_negative_expression(self) -> None:
        state = classify_expression(
            {
                "mouthPressLeft": 0.20,
                "mouthPressRight": 0.20,
                "browDownLeft": 0.20,
                "browDownRight": 0.20,
            }
        )

        self.assertEqual(state.label, "negative_expression")
        self.assertGreaterEqual(state.confidence, 0.16)

    def test_mouth_shrug_with_frown_maps_to_negative_expression(self) -> None:
        state = classify_expression(
            {
                "mouthShrugLower": 0.20,
                "mouthFrownLeft": 0.20,
                "mouthFrownRight": 0.20,
            }
        )

        self.assertEqual(state.label, "negative_expression")

    def test_mouth_shrug_alone_maps_to_neutral_expression(self) -> None:
        state = classify_expression({"mouthShrugLower": 0.80})

        self.assertEqual(state.label, "neutral_expression")

    def test_inner_brow_with_frown_maps_to_neutral_expression(self) -> None:
        state = classify_expression(
            {
                "browInnerUp": 0.20,
                "mouthFrownLeft": 0.20,
                "mouthFrownRight": 0.20,
            }
        )

        self.assertEqual(state.label, "neutral_expression")

    def test_mouth_pucker_alone_maps_to_neutral_expression(self) -> None:
        state = classify_expression({"mouthPucker": 0.80})

        self.assertEqual(state.label, "neutral_expression")

    def test_mouth_pucker_does_not_replace_a_second_sadness_cue(self) -> None:
        state = classify_expression(
            {
                "mouthPucker": 0.80,
                "browInnerUp": 0.20,
            }
        )

        self.assertEqual(state.label, "neutral_expression")

    def test_mouth_pucker_strengthens_coherent_sadness_cues(self) -> None:
        state = classify_expression(
            {
                "mouthPucker": 0.20,
                "mouthShrugLower": 0.17,
                "mouthFrownLeft": 0.17,
                "mouthFrownRight": 0.17,
            }
        )

        self.assertEqual(state.label, "negative_expression")
        self.assertGreater(state.confidence, 0.16)

    def test_nose_sneer_alone_maps_to_neutral_expression(self) -> None:
        state = classify_expression(
            {"noseSneerLeft": 0.80, "noseSneerRight": 0.80}
        )

        self.assertEqual(state.label, "neutral_expression")

    def test_nose_sneer_with_mouth_support_maps_to_negative_expression(self) -> None:
        state = classify_expression(
            {
                "noseSneerLeft": 0.20,
                "noseSneerRight": 0.20,
                "mouthUpperUpLeft": 0.20,
                "mouthUpperUpRight": 0.20,
            }
        )

        self.assertEqual(state.label, "negative_expression")

    def test_surprise_cues_map_to_positive_expression(self) -> None:
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

        self.assertEqual(state.label, "positive_expression")
        self.assertGreater(state.confidence, 0.6)

    def test_low_scores_map_to_neutral_expression(self) -> None:
        state = classify_expression({})

        self.assertEqual(state.label, "neutral_expression")
        self.assertGreaterEqual(state.confidence, 0.45)


if __name__ == "__main__":
    unittest.main()
