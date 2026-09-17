"""Observation-only computer-vision pipeline for Miscrits."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

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
    battle_ocr_cache_size: int = 128
    battle_layout_threshold: float = 0.50


class VisionEngine:
    """Analyze frames without issuing any game input."""

    _BATTLE_SIGNATURES = ("capture", "capture!", "abilities", "items", "it's your turn", "its your turn")
    _EXPLORATION_SIGNATURES = ("get miscrits", "campaign", "global boss", "daily quests", "clans")
    _ABILITY_NAMES = (
        "sting", "confuse", "smack", "power up", "swipe", "whip", "fireflight",
        "matchsticks", "shy smile", "bite",
    )
    _IGNORED_NAME_TEXT = {
        "capture", "capture!", "abilities", "items", "your turn", "its your turn",
        "it's your turn", "attack", "skills", "switch", "flee",
    }

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()
        self._ocr_cache: dict[tuple[str, str, int, str, str], tuple[str, ...]] = {}
        if pytesseract is not None:
            command = os.getenv("TESSERACT_CMD", "").strip()
            if command:
                pytesseract.pytesseract.tesseract_cmd = command

    def analyze(self, frame: Frame) -> VisionResult:
        image = frame.to_image()
        processed = self._preprocess(image)
        layout_score, layout_details = self._battle_layout_score(processed)
        if layout_score >= self.config.battle_layout_threshold:
            screen_type = "battle"
            screen_confidence = min(0.98, 0.72 + layout_score * 0.25)
            evidence = {"battle": layout_score, "exploration": 0.0}
        else:
            screen_type, screen_confidence, evidence = self._classify_non_battle(processed)

        if screen_type == "battle":
            ocr_items = self._battle_ocr_items(processed)
            battle = self._battle_observation(processed, ocr_items)
        else:
            ocr_items = self._ocr(processed)
            battle = None

        region_defs = BATTLE_REGIONS if screen_type == "battle" else EXPLORATION_REGIONS if screen_type == "exploration" else ()
        regions = [region.detect(processed, screen_confidence or 0.5) for region in region_defs]
        return VisionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            regions=regions,
            ocr_text=[item.text for item in ocr_items],
            ocr_items=ocr_items,
            screen_type=screen_type,
            screen_confidence=screen_confidence,
            battle=battle,
            diagnostics={
                "ocr_available": pytesseract is not None,
                "battle_layout_score": round(layout_score, 3),
                "battle_layout_details": layout_details,
                "battle_ocr_cache_entries": len(self._ocr_cache),
                "battle_regional_ocr": screen_type == "battle" and pytesseract is not None,
                "battle_evidence": round(evidence.get("battle", 0.0), 3),
                "exploration_evidence": round(evidence.get("exploration", 0.0), 3),
            },
        )

    def _preprocess(self, image: Image.Image) -> Image.Image:
        result = ImageOps.grayscale(image) if self.config.grayscale else image
        return result.filter(ImageFilter.SHARPEN) if self.config.sharpen else result

    def _ocr(self, image: Image.Image) -> list[OCRItem]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        try:
            data: Any = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 11")
        except Exception:
            return []
        result: list[OCRItem] = []
        for value, confidence, left, top, width, height in zip(
            data.get("text", []), data.get("conf", []), data.get("left", []),
            data.get("top", []), data.get("width", []), data.get("height", []),
        ):
            text = str(value).strip()
            try:
                score = float(confidence)
            except (TypeError, ValueError):
                score = -1.0
            if text and score >= self.config.ocr_min_confidence:
                result.append(OCRItem(text, score, int(left), int(top), int(width), int(height)))
        return result

    @classmethod
    def _classify_non_battle(cls, image: Image.Image) -> tuple[str, float, dict[str, float]]:
        evidence = {"battle": 0.0, "exploration": 0.0}
        if pytesseract is None:
            return "unknown", 0.0, evidence
        try:
            text = cls._normalize(pytesseract.image_to_string(image, config="--psm 11"))
        except Exception:
            return "unknown", 0.0, evidence
        exploration = sum(2.0 if token == "get miscrits" else 1.0 for token in cls._EXPLORATION_SIGNATURES if token in text)
        inventory = sum(weight for token, weight in (("inventory", 1.5), ("equipment", 1.0)) if token in text)
        evidence["exploration"] = exploration
        if exploration >= 2:
            return "exploration", min(0.98, 0.62 + exploration * 0.06), evidence
        if inventory:
            return "inventory", min(0.85, 0.60 + inventory * 0.08), evidence
        return ("exploration", 0.55, evidence) if exploration else ("unknown", 0.0, evidence)

    @staticmethod
    def _battle_layout_score(image: Image.Image) -> tuple[float, dict[str, float]]:
        gray = image.convert("L")
        width, height = gray.size
        if width < 500 or height < 400:
            return 0.0, {}

        def metrics(region) -> tuple[float, float]:
            left, top, right, bottom = region.crop_box(gray)
            sample = gray.crop((left, top, right, bottom)).resize((80, 40))
            pixels = list(sample.getdata())
            return sum(p >= 175 for p in pixels) / len(pixels), sum(pixels) / len(pixels)

        pb, _ = metrics(BATTLE_PLAYER_NAME)
        eb, _ = metrics(BATTLE_ENEMY_NAME)
        phb, _ = metrics(BATTLE_PLAYER_HP)
        ehb, _ = metrics(BATTLE_ENEMY_HP)
        action = next(r for r in BATTLE_REGIONS if r.name == "battle_actions")
        ab, am = metrics(action)
        field = gray.crop((int(width * .18), int(height * .20), int(width * .82), int(height * .70))).resize((80, 40))
        fp = list(field.getdata())
        fm = sum(fp) / len(fp)
        status = (pb + eb + phb + ehb) / 4
        score = .45 * min(1.0, (pb + eb) / .8) + .40 * min(1.0, max(0.0, (ab - .12) / .38)) + .15 * min(1.0, max(0.0, (am - fm + 5) / 45))
        return score, {"action_brightness": round(ab, 3), "status_brightness": round(status, 3), "action_mean": round(am, 1), "field_mean": round(fm, 1)}

    def _battle_ocr_items(self, image: Image.Image) -> list[OCRItem]:
        specs = (
            (BATTLE_PLAYER_NAME, "name", 7, None), (BATTLE_PLAYER_HP, "hp", 7, "0123456789/"),
            (BATTLE_ENEMY_NAME, "name", 7, None), (BATTLE_ENEMY_HP, "hp", 7, "0123456789/"),
            (BATTLE_CAPTURE_TEXT, "capture", 7, "Capture!%0123456789"), (BATTLE_TURN_REGION, "turn", 7, None),
        )
        result: list[OCRItem] = []
        for region, _, psm, whitelist in specs:
            for text in self._regional_ocr(image, region, psm=psm, whitelist=whitelist):
                result.append(self._make_item(image, region, text))
        for region in BATTLE_ABILITY_REGIONS:
            for text in self._regional_ocr(image, region, psm=7, whitelist=None):
                result.append(self._make_item(image, region, text))
        return result

    @staticmethod
    def _make_item(image: Image.Image, region, text: str) -> OCRItem:
        left, top, right, bottom = region.crop_box(image)
        return OCRItem(text, 100.0, left, top, right - left, bottom - top)

    def _battle_observation(self, image: Image.Image, items: list[OCRItem]) -> BattleObservation:
        def values(region) -> list[str]:
            left, top, right, bottom = region.crop_box(image)
            return [i.text for i in items if left <= i.left <= right and top <= i.top <= bottom]

        player_names = values(BATTLE_PLAYER_NAME)
        enemy_names = values(BATTLE_ENEMY_NAME)
        player_hp = self._find_hp(values(BATTLE_PLAYER_HP))
        enemy_hp = self._find_hp(values(BATTLE_ENEMY_HP))
        capture_text = " ".join(values(BATTLE_CAPTURE_TEXT))
        turn_text = " ".join(values(BATTLE_TURN_REGION))
        ability_text = [text for region in BATTLE_ABILITY_REGIONS for text in values(region)]

        player_name = self._find_name(player_names) or self._fallback_name(items, image, True)
        enemy_name = self._find_name(enemy_names) or self._fallback_name(items, image, False)
        player_current, player_max = self._parse_hp(player_hp)
        enemy_current, enemy_max = self._parse_hp(enemy_hp)
        all_text = self._normalize(" ".join(i.text for i in items))
        turn = "player" if re.search(r"it'?s your turn|your turn", self._normalize(turn_text) + " " + all_text) else None
        capture = self._parse_percent(capture_text) or self._parse_percent(all_text)
        abilities = self._extract_abilities(ability_text)
        return BattleObservation(
            player_name=player_name, enemy_name=enemy_name,
            player_hp_text=player_hp, enemy_hp_text=enemy_hp,
            player_hp_current=player_current, player_hp_max=player_max,
            enemy_hp_current=enemy_current, enemy_hp_max=enemy_max,
            turn=turn, capture_percent=capture, abilities=abilities,
            status_text=tuple(dict.fromkeys(player_names + enemy_names)),
            diagnostics={"player_name_ocr": tuple(player_names), "enemy_name_ocr": tuple(enemy_names), "player_hp_ocr": tuple(values(BATTLE_PLAYER_HP)), "enemy_hp_ocr": tuple(values(BATTLE_ENEMY_HP)), "capture_ocr": tuple(values(BATTLE_CAPTURE_TEXT)), "turn_ocr": tuple(values(BATTLE_TURN_REGION)), "ability_ocr": tuple(ability_text)},
        )

    def _regional_ocr(self, image: Image.Image, region, *, psm: int, whitelist: str | None) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        left, top, right, bottom = region.crop_box(image)
        if right <= left or bottom <= top:
            return []
        base = image.crop((left, top, right, bottom)).convert("L")
        enlarged = base.resize((base.width * self.config.battle_ocr_scale, base.height * self.config.battle_ocr_scale), Image.Resampling.LANCZOS)
        sharp = ImageEnhance.Sharpness(ImageEnhance.Contrast(enlarged).enhance(1.8)).enhance(2.0)
        variants = (("sharp", sharp), ("threshold", sharp.point(lambda p: 255 if p >= 150 else 0)))
        for variant_name, variant in variants:
            key = (region.name, hashlib.sha1(variant.tobytes()).hexdigest(), psm, whitelist or "", variant_name)
            if key in self._ocr_cache:
                result = list(self._ocr_cache[key])
            else:
                config = f"--psm {psm}"
                if whitelist:
                    config += f" -c tessedit_char_whitelist={whitelist}"
                try:
                    raw = pytesseract.image_to_string(variant, config=config)
                except Exception:
                    result = []
                else:
                    result = [line.strip() for line in raw.splitlines() if line.strip()]
                if len(self._ocr_cache) >= self.config.battle_ocr_cache_size:
                    self._ocr_cache.pop(next(iter(self._ocr_cache)))
                self._ocr_cache[key] = tuple(result)
            if result:
                return list(dict.fromkeys(result))
        return []

    @staticmethod
    def _find_hp(values: list[str]) -> str | None:
        for value in values:
            match = re.search(r"(\d{1,4})\s*[/\\|]\s*(\d{1,4})", value)
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
        current, maximum = map(int, match.groups())
        return (current, maximum) if maximum > 0 and current <= maximum else (None, None)

    @staticmethod
    def _parse_percent(text: str) -> int | None:
        matches = re.findall(r"(?:capture!?\s*)?(\d{1,3})\s*%", text.casefold())
        if not matches:
            return None
        value = int(matches[-1])
        return value if value <= 100 else None

    @classmethod
    def _find_name(cls, values: list[str]) -> str | None:
        candidates = []
        for value in values:
            cleaned = re.sub(r"[^A-Za-z0-9' -]", "", value).strip()
            if cls._looks_like_name(cleaned):
                candidates.append(cleaned)
        return max(candidates, key=len) if candidates else None

    @classmethod
    def _looks_like_name(cls, value: str) -> bool:
        return 3 <= len(value) <= 20 and not any(c.isdigit() for c in value) and value.casefold() not in cls._IGNORED_NAME_TEXT and any(c.isalpha() for c in value)

    @classmethod
    def _fallback_name(cls, items: list[OCRItem], image: Image.Image, left_side: bool) -> str | None:
        cutoff = image.width * .50
        candidates = [i.text.strip() for i in items if i.top < image.height * .16 and ((i.left < cutoff) if left_side else (i.left >= cutoff)) and cls._looks_like_name(i.text.strip())]
        return max(candidates, key=len) if candidates else None

    @classmethod
    def _extract_abilities(cls, values: list[str]) -> tuple[str, ...]:
        found: list[str] = []
        for value in values:
            text = cls._normalize(value)
            for expected in cls._ABILITY_NAMES:
                if expected in text or SequenceMatcher(None, text, expected).ratio() >= .64:
                    label = expected.title() if expected != "power up" and expected != "shy smile" else expected.title()
                    if label not in found:
                        found.append(label)
                    continue
                words = text.split()
                if any(SequenceMatcher(None, word, expected).ratio() >= .72 for word in words if len(word) >= 4):
                    label = expected.title()
                    if label not in found:
                        found.append(label)
        return tuple(found)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%/!?' -]+", " ", value.casefold().replace("’", "'"))).strip()
