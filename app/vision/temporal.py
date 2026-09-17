"""Temporal validation for observation-only vision results."""

from __future__ import annotations

from dataclasses import dataclass

from .models import BattleObservation, VisionResult


@dataclass(frozen=True)
class StabilityConfig:
    required_frames: int = 3
    max_gap_seconds: float = 2.0


@dataclass(frozen=True)
class StableBattleObservation:
    """A battle observation promoted after repeated compatible frames."""

    observation: BattleObservation
    stable_frames: int
    confidence: float
    fields_stable: tuple[str, ...]


class VisionStabilityTracker:
    """Require repeated compatible observations before reporting them as stable."""

    _FIELDS = (
        "player_name",
        "enemy_name",
        "player_hp_current",
        "player_hp_max",
        "enemy_hp_current",
        "enemy_hp_max",
        "turn",
        "capture_percent",
    )

    def __init__(self, config: StabilityConfig | None = None) -> None:
        self.config = config or StabilityConfig()
        self._last: BattleObservation | None = None
        self._count = 0
        self._last_timestamp: float | None = None

    def reset(self) -> None:
        self._last = None
        self._count = 0
        self._last_timestamp = None

    def update(self, result: VisionResult) -> StableBattleObservation | None:
        """Update the tracker and return a promoted stable observation, if any."""
        if result.screen_type != "battle" or result.battle is None:
            self.reset()
            return None

        observation = result.battle
        if self._last_timestamp is not None and result.timestamp - self._last_timestamp > self.config.max_gap_seconds:
            self._count = 0
            self._last = None

        compatible_fields = tuple(
            field for field in self._FIELDS
            if self._compatible(getattr(self._last, field, None), getattr(observation, field, None))
        ) if self._last is not None else ()

        if self._last is not None and compatible_fields:
            self._count += 1
        else:
            self._count = 1

        self._last = observation
        self._last_timestamp = result.timestamp

        if self._count < self.config.required_frames:
            return None

        confidence = min(0.99, 0.55 + 0.12 * min(self._count, 4) + 0.05 * len(compatible_fields))
        return StableBattleObservation(
            observation=observation,
            stable_frames=self._count,
            confidence=confidence,
            fields_stable=compatible_fields,
        )

    @staticmethod
    def _compatible(previous: object, current: object) -> bool:
        if previous is None or current is None:
            return previous == current or previous is None or current is None
        if isinstance(previous, int) and isinstance(current, int):
            # HP/capture values can move between frames. A small change is not
            # evidence of a broken track, so allow normal visual updates.
            return abs(previous - current) <= max(2, round(max(abs(previous), abs(current)) * 0.10))
        return previous == current
