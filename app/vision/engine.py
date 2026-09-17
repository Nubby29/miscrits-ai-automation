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
    battle_ocr_scale: int = 6
    battle_ocr_cache_size: int = 256
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
            # Keep the original-color frame for regional OCR. The tiny HUD text
            # has colored outlines/shadows that can be lost by global grayscale.
            ocr_items = self._battle_ocr_items(image)
            battle = self._battle_observation(image, ocr_items)
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
        """Run several small, specialized OCR passes instead of one generic pass."""
        specs = (
            (BATTLE_PLAYER_NAME, "name"),
            (BATTLE_PLAYER_HP, "hp"),
            (BATTLE_ENEMY_NAME, "name"),
            (BATTLE_ENEMY_HP, "hp"),
            (BATTLE_CAPTURE_TEXT, "capture"),
            (BATTLE_TURN_REGION, "turn"),
        )
        result: list[OCRItem] = []
        for region, kind in specs:
            if kind == "name":
                passes = ((7, None), (8, None), (13, None))
            elif kind == "hp":
                passes = ((7, "0123456789/"), (8, "0123456789/"), (13, "0123456789/"))
            elif kind == "capture":
                passes = ((7, "Capture%0123456789"), (8, "Capture%0123456789"), (13, "0123456789%"))
            else:
                passes = ((7, None), (8, None), (13, None))
            for psm, whitelist in passes:
                texts = self._regional_ocr(image, region, psm=psm, whitelist=whitelist)
                for text in texts:
                    result.append(self._make_item(image, region, text))

        # OCR each ability button independently. The inner text crop excludes
        # the circular element icon, which was a frequent source of false OCR.
        for index, region in enumerate(BATTLE_ABILITY_REGIONS):
            for psm, whitelist in ((7, None), (8, None), (13, None)):
                for text in self._regional_ocr(image, region, psm=psm, whitelist=whitelist, inner=True):
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
        player_hp_values = values(BATTLE_PLAYER_HP)
        enemy_hp_values = values(BATTLE_ENEMY_HP)
        player_hp = self._find_hp(player_hp_values)
        enemy_hp = self._find_hp(enemy_hp_values)
        capture_values = values(BATTLE_CAPTURE_TEXT)
        turn_values = values(BATTLE_TURN_REGION)
        ability_values = [values(region) for region in BATTLE_ABILITY_REGIONS]
        ability_text = [text for slot in ability_values for text in slot]

        player_name = self._best_name(player_names)
        enemy_name = self._best_name(enemy_names)
        player_current, player_max = self._parse_hp(player_hp)
        enemy_current, enemy_max = self._parse_hp(enemy_hp)
        turn_text = self._normalize(" ".join(turn_values))
        all_text = self._normalize(" ".join(i.text for i in items))
        turn = "player" if re.search(r"it'?s\s+your\s+turn|your\s+turn", turn_text + " " + all_text) else None

        # Capture is now parsed from its dedicated crop only. Falling back to
        # every battle OCR token could accidentally interpret an HP value such
        # as 86/86 as a capture percentage.
        capture = self._parse_capture_percent(capture_values)
        ability_slots = tuple(self._best_ability(slot) for slot in ability_values)
        abilities = tuple(label for label in ability_slots if label is not None)

        return BattleObservation(
            player_name=player_name,
            enemy_name=enemy_name,
            player_hp_text=player_hp,
            enemy_hp_text=enemy_hp,
            player_hp_current=player_current,
            player_hp_max=player_max,
            enemy_hp_current=enemy_current,
            enemy_hp_max=enemy_max,
            turn=turn,
            capture_percent=capture,
            abilities=abilities,
            ability_slots=ability_slots,
            status_text=tuple(dict.fromkeys(player_names + enemy_names)),
            diagnostics={
                "player_name_ocr": tuple(dict.fromkeys(player_names)),
                "enemy_name_ocr": tuple(dict.fromkeys(enemy_names)),
                "player_hp_ocr": tuple(dict.fromkeys(player_hp_values)),
                "enemy_hp_ocr": tuple(dict.fromkeys(enemy_hp_values)),
                "capture_ocr": tuple(dict.fromkeys(capture_values)),
                "turn_ocr": tuple(dict.fromkeys(turn_values)),
                "ability_ocr": tuple(dict.fromkeys(ability_text)),
                "ability_slots_ocr": tuple(tuple(dict.fromkeys(slot)) for slot in ability_values),
            },
        )

    def _regional_ocr(self, image: Image.Image, region, *, psm: int, whitelist: str | None, inner: bool = False) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        left, top, right, bottom = region.crop_box(image)
        if inner:
            # Preserve the right-hand text while trimming the icon/left padding.
            trim = max(1, round((right - left) * 0.22))
            left += trim
            top += max(1, round((bottom - top) * 0.08))
            bottom -= max(1, round((bottom - top) * 0.08))
        if right <= left or bottom <= top:
            return []
        base = image.crop((left, top, right, bottom)).convert("L")
        scale = max(4, self.config.battle_ocr_scale)
        enlarged = base.resize((base.width * scale, base.height * scale), Image.Resampling.LANCZOS)
        contrast = ImageEnhance.Contrast(enlarged).enhance(2.5)
        sharp = ImageEnhance.Sharpness(contrast).enhance(3.0)
        threshold = sharp.point(lambda p: 255 if p >= 135 else 0)
        mid_threshold = sharp.point(lambda p: 255 if p >= 160 else 0)
        high_threshold = sharp.point(lambda p: 255 if p >= 190 else 0)
        variants = (
            ("sharp", sharp),
            ("threshold", threshold),
            ("mid_threshold", mid_threshold),
            ("high_threshold", high_threshold),
        )
        found: list[str] = []
        for variant_name, variant in variants:
            key = (region.name + ("_inner" if inner else ""), hashlib.sha1(variant.tobytes()).hexdigest(), psm, whitelist or "", variant_name)
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
            found.extend(result)
        return list(dict.fromkeys(found))

    @staticmethod
    def _find_hp(values: list[str]) -> str | None:
        for value in values:
            normalized = value.replace("\\", "/").replace("|", "/").replace("I", "1").replace("l", "1")
            match = re.search(r"(\d{1,4})\s*/\s*(\d{1,4})", normalized)
            if match:
                current, maximum = map(int, match.groups())
                if maximum > 0 and current <= maximum:
                    return f"{current}/{maximum}"
        numbers: list[int] = []
        for value in values:
            numbers.extend(int(n) for n in re.findall(r"\d{1,4}", value))
        if len(numbers) >= 2:
            for current, maximum in zip(numbers, numbers[1:]):
                if maximum > 0 and current <= maximum and maximum <= 9999:
                    return f"{current}/{maximum}"
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
        if matches:
            value = int(matches[-1])
            return value if value <= 100 else None
        numbers = [int(n) for n in re.findall(r"\b\d{1,3}\b", text)]
        for value in reversed(numbers):
            if 0 <= value <= 100:
                return value
        return None

    @classmethod
    def _parse_capture_percent(cls, values: list[str]) -> int | None:
        """Parse capture percentage using only the dedicated capture crop."""
        normalized = [cls._normalize(value) for value in values]
        for text in normalized:
            match = re.search(r"(\d{1,3})\s*%", text)
            if match:
                value = int(match.group(1))
                if 0 <= value <= 100:
                    return value
        candidates: list[int] = []
        for text in normalized:
            # OCR may lose '%' but retain the number next to Capture.
            if "capture" in text or "captur" in text:
                candidates.extend(int(n) for n in re.findall(r"\b\d{1,3}\b", text))
        candidates = [value for value in candidates if 0 <= value <= 100]
        return candidates[-1] if candidates else None

    @classmethod
    def _best_name(cls, values: list[str]) -> str | None:
        candidates: list[str] = []
        for value in values:
            cleaned = re.sub(r"[^A-Za-z0-9' -]", "", value).strip()
            cleaned = re.sub(r"\s+", " ", cleaned)
            if cls._looks_like_name(cleaned):
                candidates.append(cleaned)
        if not candidates:
            return None
        return max(candidates, key=lambda text: (sum(c.isalpha() for c in text), len(text)))

    @classmethod
    def _looks_like_name(cls, value: str) -> bool:
        return (
            3 <= len(value) <= 20
            and not any(c.isdigit() for c in value)
            and value.casefold() not in cls._IGNORED_NAME_TEXT
            and any(c.isalpha() for c in value)
        )

    @classmethod
    def _match_ability(cls, value: str) -> tuple[str | None, float]:
        text = cls._normalize(value)
        if not text:
            return None, 0.0
        best_name: str | None = None
        best_score = 0.0
        for expected in cls._ABILITY_NAMES:
            score = SequenceMatcher(None, text, expected).ratio()
            # Compare individual OCR words too; this handles errors such as
            # "MAtchstlcks" without allowing arbitrary short text to match.
            for word in text.split():
                if len(word) >= 4:
                    score = max(score, SequenceMatcher(None, word, expected).ratio())
            if expected in text:
                score = 1.0
            if expected == "power up" and ("power" in text and "up" in text):
                score = 1.0
            if score > best_score:
                best_name, best_score = expected.title(), score
        return (best_name, best_score) if best_score >= 0.62 else (None, best_score)

    @classmethod
    def _best_ability(cls, values: list[str]) -> str | None:
        best_name: str | None = None
        best_score = 0.0
        for value in values:
            name, score = cls._match_ability(value)
            if score > best_score:
                best_name, best_score = name, score
        return best_name

    @classmethod
    def _extract_abilities(cls, values: list[str]) -> tuple[str, ...]:
        found: list[str] = []
        for value in values:
            name, score = cls._match_ability(value)
            if name and score >= 0.62 and name not in found:
                found.append(name)
        return tuple(found)

    @staticmethod
    def _normalize(value: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9%/!?' -]+", " ", value.casefold().replace("’", "'"))).strip()
