"""Observation-only computer-vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from ..capture import Frame
from .models import DetectedRegion, VisionResult
from .regions import EXPLORATION_REGIONS

try:
    import pytesseract
except ImportError:
    pytesseract = None


@dataclass(frozen=True)
class VisionConfig:
    grayscale: bool = True
    sharpen: bool = True
    enable_ocr: bool = True


class VisionEngine:
    """Analyze captured frames without issuing any game input."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    def analyze(self, frame: Frame) -> VisionResult:
        image = frame.to_image()
        processed = self._preprocess(image)
        ocr_text = self._ocr(processed)
        screen_type = self._classify_screen(ocr_text)
        regions = self._regions(processed, screen_type)
        return VisionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            regions=regions,
            ocr_text=ocr_text,
            screen_type=screen_type,
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

    def _ocr(self, image: Image.Image) -> list[str]:
        if not self.config.enable_ocr or pytesseract is None:
            return []
        try:
            data: Any = pytesseract.image_to_data(
                image, output_type=pytesseract.Output.DICT, config="--psm 11"
            )
            text: list[str] = []
            for value, confidence in zip(data.get("text", []), data.get("conf", [])):
                value = str(value).strip()
                try:
                    score = float(confidence)
                except (TypeError, ValueError):
                    score = -1
                if value and score >= 35:
                    text.append(value)
            return text
        except Exception:
            return []

    @staticmethod
    def _classify_screen(ocr_text: list[str]) -> str:
        normalized = {text.casefold().strip() for text in ocr_text}
        if {"get miscrits", "campaign", "global boss"} & normalized:
            return "exploration"
        if {"attack", "skills", "switch", "flee"} & normalized:
            return "battle"
        if {"inventory", "items", "equipment"} & normalized:
            return "inventory"
        if {"train", "my miscrits"} & normalized:
            return "menu"
        return "unknown"

    @staticmethod
    def _regions(image: Image.Image, screen_type: str) -> list[DetectedRegion]:
        if screen_type != "exploration":
            return []
        return [region.detect(image) for region in EXPLORATION_REGIONS]
