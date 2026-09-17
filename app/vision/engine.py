"""Observation-only computer-vision pipeline for Miscrits."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

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
    battle_ocr_scale: int = 3


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
        battle = self._battle_observation(processed, ocr_items) if screen_type == "battle" else None
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
                "battle_regional_ocr": bool(screen_type == "battle" and pytesseract is not None),
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
        text = value.casefold().strip().replace("’", "'")
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
        battle_score += min(2.0, cls._fuzzy_battle_hits(items) * 0.5)
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

    def _battle_observation(self, image: Image.Image, global_items: list[OCRItem]) -> BattleObservation:
        """Extract battle state using dedicated UI crops with global OCR fallback."""
        region_map = {region.name: region for region in BATTLE_REGIONS}

        player_text = self._regional_ocr(image, region_map["player_status"], psm=6)
        enemy_text = self._regional_ocr(image, region_map["enemy_status"], psm=6)
        player_hp_ocr = self._regional_ocr(
            image, region_map["player_status"], psm=7, whitelist="0123456789/ "
        )
        enemy_hp_ocr = self._regional_ocr(
            image, region_map["enemy_status"], psm=7, whitelist="0123456789/ "
        )
        capture_text = self._regional_ocr(
            image, region_map["capture_status"], psm=6, whitelist="Capture!%0123456789 "
        )
        action_text = self._regional_ocr(image, region_map["battle_actions"], psm=6)

        player_hp_text = self._find_hp(player_hp_ocr) or self._find_hp(player_text)
        enemy_hp_text = self._find_hp(enemy_hp_ocr) or self._find_hp(enemy_text)
        player_name = self._find_name(player_text)
        enemy_name = self._find_name(enemy_text)

        if not player_name or not enemy_name:
            global_names = [
                item for item in global_items
                if item.top < int(image.height * 0.22) and self._looks_like_name(item.text)
            ]
            if not player_name:
                left = [item for item in global_names if item.left < image.width * 0.55]
                player_name = left[0].text if left else None
            if not enemy_name:
                right = [item for item in global_names if item.left >= image.width * 0.45]
                enemy_name = right[0].text if right else None

        global_hp = [item.text for item in global_items if self._find_hp([item.text])]
        player_hp_text = player_hp_text or (global_hp[0] if global_hp else None)
        enemy_hp_text = enemy_hp_text or (global_hp[1] if len(global_hp) > 1 else None)

        joined = self._joined_ocr(global_items)
        turn = "player" if re.search(r"it'?s\s+your\s+turn|your\s+turn", joined) else None

        capture_joined = " ".join(self._normalize_text(text) for text in capture_text)
        capture = self._parse_percent(capture_joined) or self._parse_percent(joined)

        action_joined = " ".join(self._normalize_text(text) for text in action_text)
        abilities = self._match_abilities(action_joined) or self._match_abilities(joined)

        player_current, player_max = self._parse_hp(player_hp_text)
        enemy_current, enemy_max = self._parse_hp(enemy_hp_text)
        return BattleObservation(
            player_name=player_name,
            enemy_name=enemy_name,
            player_hp_text=player_hp_text,
            enemy_hp_text=enemy_hp_text,
            player_hp_current=player_current,
            player_hp_max=player_max,
            enemy_hp_current=enemy_current,
            enemy_hp_max=enemy_max,
            turn=turn,
            capture_percent=capture,
            abilities=abilities,
            status_text=tuple(dict.fromkeys((*player_text, *enemy_text))),
            diagnostics={
                "player_status_ocr": tuple(player_text),
                "enemy_status_ocr": tuple(enemy_text),
                "player_hp_ocr": tuple(player_hp_ocr),
                "enemy_hp_ocr": tuple(enemy_hp_ocr),
                "capture_ocr": tuple(capture_text),
                "action_ocr": tuple(action_text),
            },
        )

    def _regional_ocr(
        self,
        image: Image.Image,
        region,
        *,
        psm: int,
        whitelist: str | None = None,
    ) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        left, top, right, bottom = region.crop_box(image)
        if right <= left or bottom <= top:
            return []
        crop = image.crop((left, top, right, bottom)).convert("L")
        scale = max(1, self.config.battle_ocr_scale)
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Contrast(crop).enhance(1.35)
        crop = ImageEnhance.Sharpness(crop).enhance(1.5)
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        try:
            text = pytesseract.image_to_string(crop, config=config)
        except Exception:
            return []
        return [line.strip() for line in text.splitlines() if line.strip()]

    @staticmethod
    def _find_hp(values: Iterable[str]) -> str | None:
        for value in values:
            match = re.search(r"(\d{1,4})\s*/\s*(\d{1,4})", value)
            if match:
                return f"{match.group(1)}/{match.group(2)}"
        return None

    @staticmethod
    def _parse_hp(value: str | None) -> tuple[int | None, int | None]:
        if not value:
            return None, None
        match = re.fullmatch(r"(\d{1,4})/(\d{1,4})", value.replace(" ", ""))
        if not match:
            return None, None
        current, maximum = int(match.group(1)), int(match.group(2))
        if current > maximum or maximum == 0:
            return None, None
        return current, maximum

    @staticmethod
    def _parse_percent(text: str) -> int | None:
        matches = re.findall(r"(?:capture!?\s*)?(\d{1,3})\s*%", text)
        if not matches:
            return None
        value = int(matches[-1])
        return value if 0 <= value <= 100 else None

    @classmethod
    def _find_name(cls, values: Iterable[str]) -> str | None:
        for value in values:
            cleaned = re.sub(r"[^A-Za-z0-9' -]", "", value).strip()
            if cls._looks_like_name(cleaned):
                return cleaned
        return None

    @staticmethod
    def _looks_like_name(value: str) -> bool:
        if not 3 <= len(value) <= 20 or any(char.isdigit() for char in value):
            return False
        lowered = value.casefold()
        ignored = {
            "capture", "abilities", "items", "fireflight", "confuse", "smack", "power up",
            "your turn", "its your turn", "it's your turn",
        }
        return lowered not in ignored and any(char.isalpha() for char in value)

    @classmethod
    def _match_abilities(cls, text: str) -> tuple[str, ...]:
        found: list[str] = []
        for expected in cls._ABILITY_NAMES:
            if expected in text:
                found.append(expected.title() if expected != "power up" else "Power Up")
                continue
            words = text.split()
            if any(SequenceMatcher(None, word, expected).ratio() >= 0.72 for word in words if len(word) >= 4):
                found.append(expected.title() if expected != "power up" else "Power Up")
        return tuple(found)

    @staticmethod
    def _regions(image: Image.Image, screen_type: str) -> list[DetectedRegion]:
        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        return [region.detect(image) for region in region_defs]
