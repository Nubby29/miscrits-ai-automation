from __future__ import annotations

import unittest

from PIL import Image

from app.vision.engine import VisionEngine
from app.vision.hp_analysis import PLAYER_HP_BAR_CANDIDATE, estimate_hp_bar
from app.vision.models import BattleObservation, VisionResult
from app.vision.temporal import StabilityConfig, VisionStabilityTracker


class HPBarTests(unittest.TestCase):
    def test_estimates_synthetic_half_filled_bar(self) -> None:
        image = Image.new("RGB", (1000, 700), (35, 35, 35))
        left, top, right, bottom = PLAYER_HP_BAR_CANDIDATE.crop_box(image)
        midpoint = left + (right - left) // 2
        for x in range(left, midpoint):
            for y in range(top, bottom):
                image.putpixel((x, y), (55, 210, 85))

        estimate = estimate_hp_bar(image, PLAYER_HP_BAR_CANDIDATE)
        self.assertIsNotNone(estimate.percent)
        self.assertGreaterEqual(estimate.percent or 0, 45)
        self.assertLessEqual(estimate.percent or 100, 55)
        self.assertGreater(estimate.confidence, 0.5)

    def test_empty_region_does_not_report_fake_hp(self) -> None:
        image = Image.new("RGB", (1000, 700), (35, 35, 35))
        estimate = estimate_hp_bar(image, PLAYER_HP_BAR_CANDIDATE)
        self.assertIsNone(estimate.percent)


class BattleParsingTests(unittest.TestCase):
    def test_capture_parser_prefers_dedicated_capture_text(self) -> None:
        engine = VisionEngine()
        self.assertEqual(engine._parse_capture_percent(["Capture! 33%"]), 33)
        self.assertEqual(engine._parse_capture_percent(["Capture 34"]), 34)
        # HP values must never become capture values merely because they are
        # present elsewhere in the battle OCR result.
        self.assertIsNone(engine._parse_capture_percent(["86/86", "51/53"]))

    def test_ability_matching_handles_ocr_noise(self) -> None:
        engine = VisionEngine()
        expected = {
            "Matchsticks": "Matchsticks",
            "SHY SMILE": "Shy Smile",
            "BITE": "Bite",
            "P0WER UP": "Power Up",
        }
        for raw, label in expected.items():
            matched, score = engine._match_ability(raw)
            self.assertEqual(matched, label, raw)
            self.assertGreaterEqual(score, 0.62)

    def test_ability_slots_preserve_button_order(self) -> None:
        engine = VisionEngine()
        values = [
            ["Matchsticks"],
            ["Shy Smile"],
            ["Bite"],
            ["Power Up"],
        ]
        slots = tuple(engine._best_ability(slot) for slot in values)
        self.assertEqual(slots, ("Matchsticks", "Shy Smile", "Bite", "Power Up"))


class StabilityTests(unittest.TestCase):
    def _result(self, timestamp: float, hp: int) -> VisionResult:
        battle = BattleObservation(
            player_name="Chimney",
            enemy_name="Prawnja",
            player_hp_current=hp,
            player_hp_max=59,
            enemy_hp_current=40,
            enemy_hp_max=59,
            turn="player",
        )
        return VisionResult(
            frame_id=int(timestamp * 10),
            timestamp=timestamp,
            width=1382,
            height=736,
            screen_type="battle",
            screen_confidence=0.9,
            battle=battle,
        )

    def test_requires_repeated_frames(self) -> None:
        tracker = VisionStabilityTracker(StabilityConfig(required_frames=3))
        self.assertIsNone(tracker.update(self._result(1.0, 50)))
        self.assertIsNone(tracker.update(self._result(1.5, 50)))
        stable = tracker.update(self._result(2.0, 50))
        self.assertIsNotNone(stable)
        self.assertEqual(stable.stable_frames, 3)
        self.assertIn("enemy_hp_current", stable.fields_stable)

    def test_non_battle_resets_tracker(self) -> None:
        tracker = VisionStabilityTracker(StabilityConfig(required_frames=2))
        self.assertIsNone(tracker.update(self._result(1.0, 50)))
        tracker.update(VisionResult(2, 1.5, 1382, 736, screen_type="exploration"))
        self.assertIsNone(tracker.update(self._result(2.0, 50)))


if __name__ == "__main__":
    unittest.main()
