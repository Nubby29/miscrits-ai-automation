"""Observation-only computer-vision pipeline for Miscrits."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from ..capture import Frame
from .models import BattleObservation, OCRItem, VisionResult
from .regions import BATTLE_REGIONS, EXPLORATION_REGIONS

try:
    import pytesseract
except ImportError:
    pytesseract = None


@dataclass(frozen=True)
class VisionConfig:
    grayscale: bool = True
    sharpen: bool = True
    enable_ocr: bool = True
    ocr_min_confidence: float = 35.0


class VisionEngine:
    """Analyze frames without issuing any game input."""

    _BATTLE_SIGNATURES = {
        "capture": 2.0,
        "capture!": 2.0,
        "abilities": 1.5,
        "items": 1.0,
        "it's your turn": 2.5,
        "its your turn": 2.5,
        "fireflight": 1.0,
        "confuse": 1.0,
        "smack": 1.0,
        "power up": 1.0,
    }
    _EXPLORATION_SIGNATURES = {
        "get miscrits": 2.0,
        "campaign": 1.0,
        "global boss": 1.0,
        "daily quests": 1.0,
        "clans": 1.0,
    }
    _ABILITY_NAMES = ("fireflight", "confuse", "smack", "power up")

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._configure_tesseract()

    @staticmethod
    def _configure_tesseract() -> None:
        if pytesseract is None:
            return
        configured = os.getenv("TESSERACT_CMD", "").strip()
        if configured:
            pytesseract.pytesseract.tesseract_cmd = configured

    def analyze(self, frame: Frame) -> VisionResult:
        image = frame.to_image()
        processed = self._preprocess(image)
        ocr_items = self._ocr(processed)
        ocr_text = [item.text for item in ocr_items]
        screen_type, screen_confidence, evidence = self._classify_screen(processed, ocr_items)
        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        regions = [region.detect(processed, screen_confidence or 0.5) for region in region_defs]
        battle = self._battle_observation(ocr_items) if screen_type == "battle" else None
        return VisionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            regions=regions,
            ocr_text=ocr_text,
            ocr_items=ocr_items,
            screen_type=screen_type,
            screen_confidence=screen_confidence,
            battle=battle,
            diagnostics={
                "preprocessed": True,
                "grayscale": self.config.grayscale,
                "sharpen": self.config.sharpen,
                "ocr_available": pytesseract is not None,
                "battle_evidence": round(evidence.get("battle", 0.0), 2),
                "exploration_evidence": round(evidence.get("exploration", 0.0), 2),
            },
        )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        result = image
        if self.config.grayscale:
            result = ImageOps.grayscale(result)
        if self.config.sharpen:
            result = result.filter(ImageFilter.SHARPEN)
        return result

    def _ocr(self, image: Image.Image) -> list[OCRItem]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        try:
            data: Any = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config="--psm 11"
            )
            items: list[OCRItem] = []
            keys = ("text", "conf", "left", "top", "width", "height")
            for values in zip(*(data.get(key, []) for key in keys)):
                value, confidence, left, top, width, height = values
                text = str(value).strip()
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    score = -1.0
                if not text or score < self.config.ocr_min_confidence:
                    continue
                items.append(OCRItem(text, score, int(left), int(top), int(width), int(height)))
            return items
        except Exception:
            return []

    @classmethod
    def _normalize_text(cls, value: str) -> str:
        text = value.casefold().strip()
        text = text.replace("’", "'")
        text = re.sub(r"[^a-z0-9%/!?' ]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _joined_ocr(cls, items: list[OCRItem]) -> str:
        return " ".join(cls._normalize_text(item.text) for item in items).strip()

    @classmethod
    def _classify_screen(cls, image: Image.Image, items: list[OCRItem]) -> tuple[str, float, dict[str, float]]:
        joined = cls._joined_ocr(items)
        battle_score = sum(weight for token, weight in cls._BATTLE_SIGNATURES.items() if token in joined)
        exploration_score = sum(weight for token, weight in cls._EXPLORATION_SIGNATURES.items() if token in joined)
        inventory_score = sum(weight for token, weight in (("inventory", 1.5), ("equipment", 1.0)) if token in joined)

        if cls._battle_layout_signal(image):
            battle_score += 2.0

        fuzzy_hits = cls._fuzzy_battle_hits(items)
        battle_score += min(2.0, fuzzy_hits * 0.5)

        evidence = {"battle": battle_score, "exploration": exploration_score}
        if battle_score >= 3.0 and battle_score >= exploration_score:
            return "battle", min(0.99, 0.55 + battle_score * 0.07), evidence
        if exploration_score >= 2.0 and exploration_score > battle_score:
            return "exploration", min(0.98, 0.62 + exploration_score * 0.06), evidence
        if inventory_score > 0 and inventory_score > battle_score:
            return "inventory", min(0.85, 0.60 + inventory_score * 0.08), evidence
        if battle_score >= 2.0:
            return "battle", min(0.90, 0.50 + battle_score * 0.06), evidence
        if exploration_score > 0:
            return "exploration", 0.55, evidence
        return "unknown", 0.0, evidence

    @classmethod
    def _fuzzy_battle_hits(cls, items: list[OCRItem]) -> int:
        hits = 0
        tokens = [token for token in cls._BATTLE_SIGNATURES if len(token) >= 4]
        for item in items:
            text = cls._normalize_text(item.text)
            if len(text) < 4:
                continue
            if any(SequenceMatcher(None, text, token).ratio() >= 0.62 for token in tokens):
                hits += 1
        return hits

    @staticmethod
    def _battle_layout_signal(image: Image.Image) -> bool:
        """Detect the broad light-colored battle action strip."""
        gray = image.convert("L")
        width, height = gray.size
        if width < 500 or height < 400:
            return False

        action = gray.crop((int(width * 0.28), int(height * 0.78), int(width * 0.78), int(height * 0.98)))
        field = gray.crop((int(width * 0.18), int(height * 0.20), int(width * 0.82), int(height * 0.70)))
        action_mean = action.resize((1, 1)).getpixel((0, 0))
        field_mean = field.resize((1, 1)).getpixel((0, 0))
        bright_density = action.point(lambda p: 255 if p >= 185 else 0).resize((1, 1)).getpixel((0, 0)) / 255
        return action_mean > field_mean + 15 or bright_density >= 0.30

    @classmethod
    def _battle_observation(cls, items: list[OCRItem]) -> BattleObservation:
        """Extract conservative, position-aware battle facts from OCR."""
        if not items:
            return BattleObservation()

        frame_width = max(item.left + item.width for item in items)
        frame_height = max(item.top + item.height for item in items)
        top_cutoff = max(180, int(frame_height * 0.28))

        left_names = [
            item.text for item in items
            if item.top < top_cutoff
            and item.left < int(frame_width * 0.55)
            and 3 <= len(item.text) <= 20
            and any(ch.isalpha() for ch in item.text)
        ]
        right_names = [
            item.text for item in items
            if item.top < top_cutoff
            and item.left >= int(frame_width * 0.45)
            and 3 <= len(item.text) <= 20
            and any(ch.isalpha() for ch in item.text)
        ]

        hp = [item.text for item in items if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", item.text)]
        if not hp:
            hp = re.findall(r"\d{1,4}\s*/\s*\d{1,4}", cls._joined_ocr(items))

        abilities: list[str] = []
        for item in items:
            text = cls._normalize_text(item.text)
            for expected in cls._ABILITY_NAMES:
                if text == expected or SequenceMatcher(None, text, expected).ratio() >= 0.62:
                    abilities.append(item.text)
                    break

        joined = cls._joined_ocr(items)
        turn = "player" if any(token in joined for token in ("it's your turn", "its your turn", "your turn")) else None
        capture = None
        matches = re.findall(r"(?:capture!?\s*)?(\d{1,3})\s*%", joined)
        if matches:
            value = int(matches[-1])
            if 0 <= value <= 100:
                capture = value

        return BattleObservation(
            player_name=left_names[0] if left_names else None,
            enemy_name=right_names[0] if right_names else None,
            player_hp_text=hp[0] if len(hp) > 0 else None,
            enemy_hp_text=hp[1] if len(hp) > 1 else None,
            turn=turn,
            capture_percent=capture,
            abilities=tuple(dict.fromkeys(abilities)),
        )
