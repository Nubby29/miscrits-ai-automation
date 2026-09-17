"""Typed results produced by the vision engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectedRegion:
    """A named rectangle detected in the captured frame."""

    name: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 1.0


@dataclass
class VisionResult:
    """Frame analysis result, intentionally independent from game actions."""

    frame_id: int
    timestamp: float
    width: int
    height: int
    regions: list[DetectedRegion] = field(default_factory=list)
    ocr_text: list[str] = field(default_factory=list)
    screen_type: str = "unknown"
    diagnostics: dict[str, object] = field(default_factory=dict)
