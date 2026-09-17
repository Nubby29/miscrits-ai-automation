"""Observation-only computer-vision pipeline for Miscrits."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from ..capture import Frame
from .models import BattleObservation, DetectedRegion, OCRItem, VisionResult
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
        screen_type, screen_confidence = self._classify_screen(processed, ocr_text)
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
    def _classify_screen(cls, image: Image.Image, ocr_text: list[str]) -> tuple[str, float]:
        joined = " ".join(ocr_text).casefold()
        battle_hits = sum(
            token in joined
            for token in ("capture", "it's your turn", "fireflight", "confuse", "smack", "power up")
        )
        exploration_hits = sum(
            token in joined
            for token in ("get miscrits", "campaign", "global boss", "daily quests", "clans")
        )
        inventory_hits = sum(token in joined for token in ("inventory", "equipment"))

        if battle_hits >= 2 or (battle_hits >= 1 and cls._battle_layout_signal(image)):
            return "battle", min(0.99, 0.70 + battle_hits * 0.08)
        if exploration_hits >= 2:
            return "exploration", min(0.98, 0.68 + exploration_hits * 0.07)
        if inventory_hits:
            return "inventory", 0.70
        if battle_hits:
            return "battle", 0.55
        if exploration_hits:
            return "exploration", 0.55
        if cls._battle_layout_signal(image):
            return "battle", 0.58
        return "unknown", 0.0

    @staticmethod
    def _battle_layout_signal(image: Image.Image) -> bool:
        """Detect the broad light-colored action strip characteristic of battle UI."""
        gray = image.convert("L")
        width, height = gray.size
        if width < 500 or height < 400:
            return False
        # The battle action strip occupies the lower-middle portion. Compare
        # brightness of that strip against the world/background area.
        action = gray.crop((int(width * 0.32), int(height * 0.74), int(width * 0.74), int(height * 0.97)))
        field = gray.crop((int(width * 0.20), int(height * 0.20), int(width * 0.80), int(height * 0.70)))
        action_mean = sum(action.resize((1, 1)).getdata()) / 1
        field_mean = sum(field.resize((1, 1)).getdata()) / 1
        return action_mean > field_mean + 18

    @staticmethod
    def _battle_observation(items: list[OCRItem]) -> BattleObservation:
        """Extract conservative battle facts from positioned OCR text."""
        def nearby_text(y_min: float, y_max: float) -> list[OCRItem]:
            return [item for item in items if y_min <= item.top <= y_max]

        names = [item.text for item in items if item.top < 180 and 3 <= len(item.text) <= 20]
        hp = [item.text for item in items if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", item.text)]
        abilities: list[str] = []
        for item in items:
            normalized = item.text.casefold().replace("0", "o")
            if normalized in {"fireflight", "confuse", "smack", "power up"}:
                abilities.append(item.text)

        joined = " ".join(item.text for item in nearby_text(0, 700)).casefold()
        turn = "player" if "it's your turn" in joined or "its your turn" in joined else None
        capture = None
        match = re.search(r"(?:capture!?\s*)?(\d{1,3})\s*%", joined)
        if match:
            capture = int(match.group(1))

        return BattleObservation(
            player_name=names[0] if len(names) > 0 else None,
            enemy_name=names[1] if len(names) > 1 else None,
            player_hp_text=hp[0] if len(hp) > 0 else None,
            enemy_hp_text=hp[1] if len(hp) > 1 else None,
            turn=turn,
            capture_percent=capture,
            abilities=tuple(dict.fromkeys(abilities)),
        )

    @staticmethod
    def _regions(image: Image.Image, screen_type: str) -> list[DetectedRegion]:
        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        return [region.detect(image) for region in region_defs]
