"""Utilities for saving and loading captured frames as vision fixtures."""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from ..capture import Frame

FIXTURE_ROOT = Path("fixtures")


def save_frame(frame: Frame, category: str = "unknown", name: str | None = None) -> Path:
    """Save a captured frame as a PNG fixture and return its path."""
    safe_category = "".join(c if c.isalnum() or c in "-_" else "_" for c in category).strip("_") or "unknown"
    directory = FIXTURE_ROOT / safe_category
    directory.mkdir(parents=True, exist_ok=True)
    filename = name or f"frame_{frame.frame_id:06d}.png"
    if not filename.lower().endswith(".png"):
        filename += ".png"
    path = directory / filename
    frame.to_image().save(path, format="PNG")
    return path


def load_fixture(path: str | Path) -> Image.Image:
    """Load a fixture image for offline vision testing."""
    return Image.open(path).convert("RGB")
