"""Typed results produced by the vision engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DetectedRegion:
    """A named rectangle detected or defined in the captured frame."""

    name: str
    left: int
    top: int
    width: int
    height: int
    confidence: float = 1.0

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


@dataclass(frozen=True)
class OCRItem:
    text: str
    confidence: float
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True)
class BattleObservation:
    """Best-effort visual observations from a battle screen."""

    player_name: str | None = None
    enemy_name: str | None = None
    player_hp_text: str | None = None
    enemy_hp_text: str | None = None
    player_hp_current: int | None = None
    player_hp_max: int | None = None
    enemy_hp_current: int | None = None
    enemy_hp_max: int | None = None
    turn: str | None = None
    capture_percent: int | None = None
    abilities: tuple[str, ...] = ()
    status_text: tuple[str, ...] = ()
    diagnostics: dict[str, object] = field(default_factory=dict)


@dataclass
class VisionResult:
    """Frame analysis result, independent from game actions."""

    frame_id: int
    timestamp: float
    width: int
    height: int
    regions: list[DetectedRegion] = field(default_factory=list)
    ocr_text: list[str] = field(default_factory=list)
    ocr_items: list[OCRItem] = field(default_factory=list)
    screen_type: str = "unknown"
    screen_confidence: float = 0.0
    battle: BattleObservation | None = None
    diagnostics: dict[str, object] = field(default_factory=dict)
