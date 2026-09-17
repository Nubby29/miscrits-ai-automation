"""Observation-only computer-vision pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageFilter, ImageOps

from ..capture import Frame
from .models import VisionResult


@dataclass(frozen=True)
class VisionConfig:
    """Configuration for inexpensive first-pass frame analysis."""

    grayscale: bool = True
    sharpen: bool = True


class VisionEngine:
    """Analyze captured frames without issuing any game input."""

    def __init__(self, config: VisionConfig | None = None) -> None:
        self.config = config or VisionConfig()

    def analyze(self, frame: Frame) -> VisionResult:
        image = self._to_image(frame)
        processed = self._preprocess(image)
        screen_type = self._classify_screen(processed)
        return VisionResult(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            width=frame.width,
            height=frame.height,
            screen_type=screen_type,
            diagnostics={
                "preprocessed": True,
                "grayscale": self.config.grayscale,
                "sharpen": self.config.sharpen,
            },
        )

    @staticmethod
    def _to_image(frame: Frame) -> Image.Image:
        pixels: Any = frame.pixels
        return Image.frombytes("RGB", (frame.width, frame.height), pixels.rgb)

    def _preprocess(self, image: Image.Image) -> Image.Image:
        result = image
        if self.config.grayscale:
            result = ImageOps.grayscale(result)
        if self.config.sharpen:
            result = result.filter(ImageFilter.SHARPEN)
        return result

    @staticmethod
    def _classify_screen(image: Image.Image) -> str:
        """Return a conservative screen class until game-specific templates exist."""
        # This deliberately avoids guessing from arbitrary screenshots.
        # Game-specific visual signatures will be added after real fixtures are available.
        _ = image
        return "unknown"
