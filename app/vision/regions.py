"""Normalized, resolution-independent regions for Miscrits screen analysis."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .models import DetectedRegion


@dataclass(frozen=True)
class NormalizedRegion:
    name: str
    x: float
    y: float
    width: float
    height: float

    def crop_box(self, image: Image.Image) -> tuple[int, int, int, int]:
        image_width, image_height = image.size
        left = round(self.x * image_width)
        top = round(self.y * image_height)
        right = round((self.x + self.width) * image_width)
        bottom = round((self.y + self.height) * image_height)
        return max(0, left), max(0, top), min(image_width, right), min(image_height, bottom)

    def detect(self, image: Image.Image, confidence: float = 1.0) -> DetectedRegion:
        left, top, right, bottom = self.crop_box(image)
        return DetectedRegion(self.name, left, top, max(0, right - left), max(0, bottom - top), confidence)


# Coordinates are normalized against the captured Miscrits client window.
# They are layout anchors, not assumptions about a particular resolution.
EXPLORATION_REGIONS = (
    NormalizedRegion("top_bar", 0.20, 0.00, 0.68, 0.15),
    NormalizedRegion("left_menu", 0.00, 0.16, 0.20, 0.60),
    NormalizedRegion("world_area", 0.18, 0.14, 0.82, 0.72),
    NormalizedRegion("party_bar", 0.18, 0.83, 0.57, 0.17),
    NormalizedRegion("bottom_actions", 0.75, 0.83, 0.25, 0.17),
)

BATTLE_REGIONS = (
    NormalizedRegion("player_status", 0.20, 0.02, 0.28, 0.14),
    NormalizedRegion("enemy_status", 0.58, 0.02, 0.25, 0.14),
    NormalizedRegion("battle_field", 0.12, 0.14, 0.76, 0.61),
    NormalizedRegion("capture_status", 0.40, 0.08, 0.20, 0.16),
    NormalizedRegion("party_switcher", 0.15, 0.30, 0.18, 0.42),
    NormalizedRegion("battle_actions", 0.34, 0.76, 0.38, 0.22),
)
