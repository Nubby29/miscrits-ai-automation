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
    battle_layout_threshold: float = 0.58


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
    _ABILITY_NAMES = (
        "sting", "confuse", "smack", "power up", "swipe", "whip", "fireflight",
    )
    _NAME_WHITELIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' -"
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

        # Battle layout is cheap and deterministic for the known Miscrits HUD.
        # Use it before OCR so a failed tiny-text probe cannot turn a real battle
        # into "unknown" and trigger an expensive full-frame OCR pass.
        layout_score, layout_details = self._battle_layout_score(processed)
        if layout_score >= self.config.battle_layout_threshold:
            screen_type = "battle"
            screen_confidence = min(0.98, 0.68 + layout_score * 0.30)
            evidence = {"battle": 4.0 * layout_score, "exploration": 0.0}
        else:
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
                "battle_layout_score": round(layout_score, 3),
                "battle_layout_details": layout_details,
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
        return cls._classify_non_battle(image, evidence)

    @classmethod
    def _classify_non_battle(cls, image: Image.Image, evidence: dict[str, float]) -> tuple[str, float, dict[str, float]]:
        if pytesseract is None:
            return "unknown", 0.0, evidence
        try:
            joined = cls._normalize_text(pytesseract.image_to_string(image, config="--psm 11"))
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
        for value in values:
            text = cls._normalize_text(value.text if isinstance(value, OCRItem) else value)
            if len(text) < 4:
                continue
            if any(SequenceMatcher(None, text, token).ratio() >= 0.62 for token in cls._BATTLE_SIGNATURES):
                hits += 1
        return hits

    @classmethod
    def _battle_layout_score(cls, image: Image.Image) -> tuple[float, dict[str, float]]:
        """Score the fixed battle HUD geometry without running OCR."""
        gray = image.convert("L")
        width, height = gray.size
        if width < 500 or height < 400:
            return 0.0, {"action_brightness": 0.0, "status_brightness": 0.0, "contrast": 0.0}

        def metrics(region) -> tuple[float, float]:
            left, top, right, bottom = region.crop_box(gray)
            crop = gray.crop((left, top, right, bottom)).resize((80, 40))
            pixels = list(crop.getdata())
            bright = sum(pixel >= 175 for pixel in pixels) / len(pixels)
            mean = sum(pixels) / len(pixels)
            return bright, mean

        player_bright, player_mean = metrics(BATTLE_PLAYER_NAME)
        enemy_bright, enemy_mean = metrics(BATTLE_ENEMY_NAME)
        player_hp_bright, _ = metrics(BATTLE_PLAYER_HP)
        enemy_hp_bright, _ = metrics(BATTLE_ENEMY_HP)
        action_region = next(region for region in BATTLE_REGIONS if region.name == "battle_actions")
        action_bright, action_mean = metrics(action_region)
        field = gray.crop((int(width * 0.18), int(height * 0.20), int(width * 0.82), int(height * 0.70))).resize((80, 40))
        field_pixels = list(field.getdata())
        field_mean = sum(field_pixels) / len(field_pixels)

        status_brightness = (player_bright + enemy_bright + player_hp_bright + enemy_hp_bright) / 4
        status_pair = min(1.0, (player_bright + enemy_bright) / 0.8)
        action_signal = min(1.0, max(0.0, (action_bright - 0.12) / 0.38))
        contrast_signal = min(1.0, max(0.0, (action_mean - field_mean + 5) / 45))
        score = 0.45 * status_pair + 0.40 * action_signal + 0.15 * contrast_signal
        return score, {
            "action_brightness": round(action_bright, 3),
            "status_brightness": round(status_brightness, 3),
            "player_status_brightness": round(player_bright, 3),
            "enemy_status_brightness": round(enemy_bright, 3),
            "action_mean": round(action_mean, 1),
            "field_mean": round(field_mean, 1),
        }

    @classmethod
    def _battle_layout_signal(cls, image: Image.Image) -> bool:
        score, _ = cls._battle_layout_score(image)
        return score >= 0.50

    def _battle_probe(self, image: Image.Image) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        return self._regional_ocr(image, BATTLE_TURN_REGION, psm=7) + self._regional_ocr(
            image, BATTLE_CAPTURE_TEXT, psm=7, whitelist="Capture!%0123456789 "
        )

    def _battle_ocr_items(self, image: Image.Image) -> list[OCRItem]:
        values: list[OCRItem] = []
        regions = (
            (BATTLE_PLAYER_NAME, "name"), (BATTLE_PLAYER_HP, "hp"),
            (BATTLE_ENEMY_NAME, "name"), (BATTLE_ENEMY_HP, "hp"),
            (BATTLE_CAPTURE_TEXT, "capture"), (BATTLE_TURN_REGION, "turn"),
        )
        for region, kind in regions:
            lines = self._regional_ocr(
                image, region, psm=7 if kind in {"name", "hp", "turn"} else 6,
                whitelist="0123456789/ " if kind == "hp" else self._NAME_WHITELIST if kind == "name" else None,
            )
            left, top, right, bottom = region.crop_box(image)
            for line in lines:
                values.append(OCRItem(line, 100.0, left, top, max(1, right - left), max(1, bottom - top)))

        for region in BATTLE_ABILITY_REGIONS:
            lines = self._regional_ocr(image, region, psm=7, whitelist=self._NAME_WHITELIST)
            left, top, right, bottom = region.crop_box(image)
            for line in lines:
                values.append(OCRItem(line, 100.0, left, top, max(1, right - left), max(1, bottom - top)))
        return values

    @staticmethod
    def _items_in_region(items: list[OCRItem], region, image: Image.Image) -> list[str]:
        left, top, right, bottom = region.crop_box(image)
        values: list[str] = []
        for item in items:
            center_x = item.left + item.width / 2
            center_y = item.top + item.height / 2
            if left <= center_x <= right and top <= center_y <= bottom:
                values.append(item.text)
        return values

    def _battle_observation(self, image: Image.Image, items: list[OCRItem]) -> BattleObservation:
        player_name_values = self._items_in_region(items, BATTLE_PLAYER_NAME, image)
        enemy_name_values = self._items_in_region(items, BATTLE_ENEMY_NAME, image)
        player_hp_values = self._items_in_region(items, BATTLE_PLAYER_HP, image)
        enemy_hp_values = self._items_in_region(items, BATTLE_ENEMY_HP, image)
        capture_values = self._items_in_region(items, BATTLE_CAPTURE_TEXT, image)
        turn_values = self._items_in_region(items, BATTLE_TURN_REGION, image)

        player_name = self._find_name(player_name_values) or self._fallback_name(items, left_side=True, image=image)
        enemy_name = self._find_name(enemy_name_values) or self._fallback_name(items, left_side=False, image=image)
        player_hp_text = self._find_hp(player_hp_values) or self._fallback_hp(items, index=0)
        enemy_hp_text = self._find_hp(enemy_hp_values) or self._fallback_hp(items, index=1)

        turn = "player" if re.search(r"it'?s\s+your\s+turn|your\s+turn", self._joined_ocr((*turn_values, *items))) else None
        capture = self._parse_percent(self._joined_ocr(capture_values)) or self._parse_percent(self._joined_ocr(items))

        abilities: list[str] = []
        for region in BATTLE_ABILITY_REGIONS:
            values = self._items_in_region(items, region, image)
            abilities.extend(self._extract_abilities(values))
        if not abilities:
            abilities = list(self._extract_abilities([item.text for item in items]))

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
            abilities=tuple(dict.fromkeys(abilities)),
            status_text=status,
            diagnostics={
                "player_name_ocr": tuple(player_name_values),
                "enemy_name_ocr": tuple(enemy_name_values),
                "player_hp_ocr": tuple(player_hp_values),
                "enemy_hp_ocr": tuple(enemy_hp_values),
                "capture_ocr": tuple(capture_values),
                "turn_ocr": tuple(turn_values),
                "ability_regions": tuple(region.name for region in BATTLE_ABILITY_REGIONS),
                "all_battle_ocr": tuple(item.text for item in items),
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
        crop = ImageEnhance.Contrast(crop).enhance(1.5)
        crop = ImageEnhance.Sharpness(crop).enhance(1.8)
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
        return value.casefold() not in cls._IGNORED_NAME_TEXT and any(char.isalpha() for char in value)

    @classmethod
    def _extract_abilities(cls, values: Iterable[str]) -> tuple[str, ...]:
        found: list[str] = []
        for value in values:
            text = cls._normalize_text(value)
            if not text:
                continue
            for expected in cls._ABILITY_NAMES:
                if expected in text or SequenceMatcher(None, text, expected).ratio() >= 0.70:
                    label = expected.title() if expected != "power up" else "Power Up"
                    if label not in found:
                        found.append(label)
                    continue
                for word in text.split():
                    if len(word) >= 4 and SequenceMatcher(None, word, expected).ratio() >= 0.72:
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
