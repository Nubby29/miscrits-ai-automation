"""Observation-only computer-vision pipeline for Miscrits."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any, Iterable

from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from ..capture import Frame
from .battle_regions import (
    BATTLE_ABILITY_REGIONS,
    BATTLE_CAPTURE_TEXT,
    BATTLE_ENEMY_HP,
    BATTLE_ENEMY_NAME,
    BATTLE_PLAYER_HP,
    BATTLE_PLAYER_NAME,
    BATTLE_TURN_REGION,
)
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
    battle_ocr_scale: int = 4
    battle_ocr_cache_size: int = 64


class VisionEngine:
    """Analyze frames without issuing any game input."""

    _BATTLE_SIGNATURES = {
        "capture": 2.0,
        "capture!": 2.0,
        "abilities": 1.5,
        "items": 1.0,
        "it's your turn": 2.5,
        "its your turn": 2.5,
    }
    _EXPLORATION_SIGNATURES = {
        "get miscrits": 2.0,
        "campaign": 1.0,
        "global boss": 1.0,
        "daily quests": 1.0,
        "clans": 1.0,
    }
    _ABILITY_NAMES = ("sting", "confuse", "smack", "power up", "swipe", "whip", "fireflight")
    _IGNORED_NAME_TEXT = {
        "capture", "capture!", "abilities", "items", "your turn", "its your turn",
        "it's your turn", "attack", "skills", "switch", "flee",
    }

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._ocr_cache: dict[tuple[str, str, int, str], tuple[str, ...]] = {}
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

        # Battle layout is visually distinctive. Use only a tiny OCR probe for
        # classification; expensive full-frame OCR is reserved for other screens.
        battle_probe = self._battle_probe(processed)
        screen_type, screen_confidence, evidence = self._classify_screen(processed, battle_probe)

        if screen_type == "battle":
            ocr_items = self._battle_ocr_items(processed)
            battle = self._battle_observation(processed, ocr_items)
        else:
            ocr_items = self._ocr(processed)
            battle = None

        ocr_text = [item.text for item in ocr_items]
        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        regions = [region.detect(processed, screen_confidence or 0.5) for region in region_defs]

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
                "battle_ocr_cache_entries": len(self._ocr_cache),
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
        text = re.sub(r"[^a-z0-9%/!?' -]+", " ", text)
        return re.sub(r"\s+", " ", text).strip()

    @classmethod
    def _joined_ocr(cls, items: Iterable[OCRItem | str]) -> str:
        values = (item.text if isinstance(item, OCRItem) else item for item in items)
        return " ".join(cls._normalize_text(value) for value in values).strip()

    @classmethod
    def _classify_screen(cls, image: Image.Image, probe: list[str]) -> tuple[str, float, dict[str, float]]:
        joined = cls._joined_ocr(probe)
        battle_score = sum(weight for token, weight in cls._BATTLE_SIGNATURES.items() if token in joined)
        exploration_score = 0.0
        if cls._battle_layout_signal(image):
            battle_score += 2.5
        battle_score += min(1.5, cls._fuzzy_battle_hits(probe) * 0.5)
        evidence = {"battle": battle_score, "exploration": exploration_score}

        if battle_score >= 2.5:
            return "battle", min(0.97, 0.66 + battle_score * 0.055), evidence

        # If the battle visual signal is absent, perform a normal full-frame OCR
        # classification. This path is deliberately skipped for battle frames.
        return cls._classify_non_battle(image, evidence)

    @classmethod
    def _classify_non_battle(cls, image: Image.Image, evidence: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        if pytesseract is None:
            return "unknown", 0.0, evidence
        try:
            data: Any = pytesseract.image_to_string(image, config="--psm 11")
            joined = cls._normalize_text(str(data))
        except Exception:
            return "unknown", 0.0, evidence
        exploration_score = sum(weight for token, weight in cls._EXPLORATION_SIGNATURES.items() if token in joined)
        inventory_score = sum(weight for token, weight in (("inventory", 1.5), ("equipment", 1.0)) if token in joined)
        evidence["exploration"] = exploration_score
        if exploration_score >= 2.0:
            return "exploration", min(0.98, 0.62 + exploration_score * 0.06), evidence
        if inventory_score > 0:
            return "inventory", min(0.85, 0.60 + inventory_score * 0.08), evidence
        if exploration_score > 0:
            return "exploration", 0.55, evidence
        return "unknown", 0.0, evidence

    @classmethod
    def _fuzzy_battle_hits(cls, values: Iterable[OCRItem | str]) -> int:
        hits = 0
        tokens = tuple(cls._BATTLE_SIGNATURES)
        for value in values:
            text = cls._normalize_text(value.text if isinstance(value, OCRItem) else value)
            if len(text) < 4:
                continue
            if any(SequenceMatcher(None, text, token).ratio() >= 0.62 for token in tokens):
                hits += 1
        return hits

    @staticmethod
    def _battle_layout_signal(image: Image.Image) -> bool:
        """Detect the bright battle action strip without OCR."""
        gray = image.convert("L")
        width, height = gray.size
        if width < 500 or height < 400:
            return False
        action = gray.crop((int(width * 0.28), int(height * 0.73), int(width * 0.80), int(height * 0.99)))
        field = gray.crop((int(width * 0.18), int(height * 0.20), int(width * 0.82), int(height * 0.70)))
        action_mean = action.resize((1, 1)).getpixel((0, 0))
        field_mean = field.resize((1, 1)).getpixel((0, 0))
        bright = sum(1 for pixel in action.resize((80, 40)).getdata() if pixel >= 185) / 3200
        return action_mean > field_mean + 12 or bright >= 0.28

    def _battle_probe(self, image: Image.Image) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        # The action strip is enough to distinguish battle from most normal UI.
        return self._regional_ocr(image, BATTLE_TURN_REGION, psm=7) + self._regional_ocr(
            image, BATTLE_CAPTURE_TEXT, psm=7, whitelist="Capture!%0123456789 "
        )

    def _battle_ocr_items(self, image: Image.Image) -> list[OCRItem]:
        """Return compact OCR items from the battle text-bearing areas."""
        values: list[OCRItem] = []
        regions = (
            (BATTLE_PLAYER_NAME, "name"),
            (BATTLE_PLAYER_HP, "hp"),
            (BATTLE_ENEMY_NAME, "name"),
            (BATTLE_ENEMY_HP, "hp"),
            (BATTLE_CAPTURE_TEXT, "capture"),
            (BATTLE_TURN_REGION, "turn"),
        )
        for region, kind in regions:
            lines = self._regional_ocr(
                image,
                region,
                psm=7 if kind in {"name", "hp", "turn"} else 6,
                whitelist="0123456789/ " if kind == "hp" else None,
            )
            left, top, right, bottom = region.crop_box(image)
            for line in lines:
                values.append(OCRItem(line, 100.0, left, top, max(1, right - left), max(1, bottom - top)))

        action_lines = self._regional_ocr(
            image,
            # The existing battle_actions region includes all four buttons.
            next(region for region in BATTLE_REGIONS if region.name == "battle_actions"),
            psm=6,
        )
        action_region = next(region for region in BATTLE_REGIONS if region.name == "battle_actions")
        left, top, right, bottom = action_region.crop_box(image)
        for line in action_lines:
            values.append(OCRItem(line, 100.0, left, top, max(1, right - left), max(1, bottom - top)))
        return values

    def _battle_observation(self, image: Image.Image, items: list[OCRItem]) -> BattleObservation:
        """Extract battle state from isolated, cached HUD crops."""
        player_name_values = self._regional_ocr(image, BATTLE_PLAYER_NAME, psm=7)
        enemy_name_values = self._regional_ocr(image, BATTLE_ENEMY_NAME, psm=7)
        player_hp_values = self._regional_ocr(image, BATTLE_PLAYER_HP, psm=7, whitelist="0123456789/ ")
        enemy_hp_values = self._regional_ocr(image, BATTLE_ENEMY_HP, psm=7, whitelist="0123456789/ ")
        capture_values = self._regional_ocr(image, BATTLE_CAPTURE_TEXT, psm=7, whitelist="Capture!%0123456789 ")
        turn_values = self._regional_ocr(image, BATTLE_TURN_REGION, psm=7)
        action_region = next(region for region in BATTLE_REGIONS if region.name == "battle_actions")
        action_values = self._regional_ocr(image, action_region, psm=6)

        player_name = self._find_name(player_name_values)
        enemy_name = self._find_name(enemy_name_values)
        player_hp_text = self._find_hp(player_hp_values)
        enemy_hp_text = self._find_hp(enemy_hp_values)

        # Use global OCR coordinates only as a fallback for a missed tiny crop.
        if not player_name:
            player_name = self._fallback_name(items, left_side=True, image=image)
        if not enemy_name:
            enemy_name = self._fallback_name(items, left_side=False, image=image)
        if not player_hp_text:
            player_hp_text = self._fallback_hp(items, index=0)
        if not enemy_hp_text:
            player_hp_candidates = [self._find_hp([item.text]) for item in items]
            hp = [value for value in player_hp_candidates if value]
            enemy_hp_text = hp[1] if len(hp) > 1 else None

        joined = self._joined_ocr((*turn_values, *capture_values, *action_values, *items))
        turn = "player" if re.search(r"it'?s\s+your\s+turn|your\s+turn", joined) else None
        capture = self._parse_percent(self._joined_ocr(capture_values)) or self._parse_percent(joined)
        abilities = self._extract_abilities(action_values)

        player_current, player_max = self._parse_hp(player_hp_text)
        enemy_current, enemy_max = self._parse_hp(enemy_hp_text)
        status = tuple(dict.fromkeys((*player_name_values, *enemy_name_values, *player_hp_values, *enemy_hp_values)))
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
            status_text=status,
            diagnostics={
                "player_name_ocr": tuple(player_name_values),
                "enemy_name_ocr": tuple(enemy_name_values),
                "player_hp_ocr": tuple(player_hp_values),
                "enemy_hp_ocr": tuple(enemy_hp_values),
                "capture_ocr": tuple(capture_values),
                "turn_ocr": tuple(turn_values),
                "action_ocr": tuple(action_values),
                "ability_regions": tuple(region.name for region in BATTLE_ABILITY_REGIONS),
            },
        )

    def _regional_ocr(self, image: Image.Image, region, *, psm: int, whitelist: str | None = None) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        left, top, right, bottom = region.crop_box(image)
        if right <= left or bottom <= top:
            return []
        crop = image.crop((left, top, right, bottom)).convert("L")
        scale = max(1, self.config.battle_ocr_scale)
        crop = crop.resize((crop.width * scale, crop.height * scale), Image.Resampling.LANCZOS)
        crop = ImageEnhance.Contrast(crop).enhance(1.45)
        crop = ImageEnhance.Sharpness(crop).enhance(1.7)
        # A light threshold variant makes the outlined HUD text easier to read.
        if whitelist is not None:
            crop = crop.point(lambda p: 255 if p >= 150 else 0)
        digest = hashlib.sha1(crop.tobytes()).hexdigest()
        key = (region.name, digest, psm, whitelist or "")
        cached = self._ocr_cache.get(key)
        if cached is not None:
            return list(cached)
        config = f"--psm {psm}"
        if whitelist:
            config += f" -c tessedit_char_whitelist={whitelist}"
        try:
            text = pytesseract.image_to_string(crop, config=config)
        except Exception:
            return []
        values = tuple(line.strip() for line in text.splitlines() if line.strip())
        if len(self._ocr_cache) >= self.config.battle_ocr_cache_size:
            self._ocr_cache.pop(next(iter(self._ocr_cache)))
        self._ocr_cache[key] = values
        return list(values)

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
        if maximum == 0 or current > maximum:
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
        candidates: list[str] = []
        for value in values:
            cleaned = re.sub(r"[^A-Za-z0-9' -]", "", value).strip()
            if cls._looks_like_name(cleaned):
                candidates.append(cleaned)
        return max(candidates, key=len) if candidates else None

    @classmethod
    def _looks_like_name(cls, value: str) -> bool:
        if not 3 <= len(value) <= 20 or any(char.isdigit() for char in value):
            return False
        lowered = value.casefold()
        return lowered not in cls._IGNORED_NAME_TEXT and any(char.isalpha() for char in value)

    @classmethod
    def _extract_abilities(cls, values: Iterable[str]) -> tuple[str, ...]:
        found: list[str] = []
        for value in values:
            text = cls._normalize_text(value)
            if not text:
                continue
            for expected in cls._ABILITY_NAMES:
                if text == expected or SequenceMatcher(None, text, expected).ratio() >= 0.70:
                    label = expected.title() if expected != "power up" else "Power Up"
                    if label not in found:
                        found.append(label)
                    break
        return tuple(found)

    @classmethod
    def _fallback_name(cls, items: list[OCRItem], *, left_side: bool, image: Image.Image) -> str | None:
        cutoff = image.width * (0.55 if left_side else 0.45)
        candidates = [
            item for item in items
            if item.top < image.height * 0.16
            and ((item.left < cutoff) if left_side else (item.left >= cutoff))
            and cls._looks_like_name(item.text)
        ]
        return max((item.text.strip() for item in candidates), key=len, default=None)

    @classmethod
    def _fallback_hp(cls, items: list[OCRItem], *, index: int) -> str | None:
        values = [cls._find_hp([item.text]) for item in items]
        hp = [value for value in values if value]
        return hp[index] if index < len(hp) else None

    @staticmethod
    def _regions(image: Image.Image, screen_type: str) -> list[DetectedRegion]:
        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        return [region.detect(image) for region in region_defs]
