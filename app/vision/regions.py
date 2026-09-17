"""Normalized, resolution-independent regions for game-screen analysis."""

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
        return (
            max(0, left),
            max(0, top),
            min(image_width, right),
            min(image_height, bottom),
        )

    def detect(self, image: Image.Image) -> DetectedRegion:
        left, top, right, bottom = self.crop_box(image)
        return DetectedRegion(
            name=self.name,
            left=left,
            top=top,
            width=max(0, right - left),
            height=max(0, bottom - top),
        )


# Initial exploration layout regions. These are intentionally broad and can be
# tuned after additional real-game screenshots are collected.
EXPLORATION_REGIONS: tuple[NormalizedRegion, ...] = (
    NormalizedRegion("top_bar", 0.25, 0.0, 0.55, 0.14),
    NormalizedRegion("left_menu", 0.0, 0.16, 0.18, 0.62),
    NormalizedRegion("world_area", 0.18, 0.14, 0.82, 0.73),
    NormalizedRegion("party_bar", 0.18, 0.84, 0.58, 0.16),
    NormalizedRegion("bottom_actions", 0.76, 0.84, 0.24, 0.16),
)
