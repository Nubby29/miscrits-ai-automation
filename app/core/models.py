"""Domain models used by the automation engine."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class AutomationState(str, Enum):
    IDLE = "idle"
    STARTING = "starting"
    OBSERVING = "observing"
    DECIDING = "deciding"
    ACTING = "acting"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"


class Point(BaseModel):
    # Desktop coordinates may be negative when a monitor is positioned to the
    # left or above the primary Windows display.
    x: int
    y: int


class Region(BaseModel):
    # Windows virtual-desktop coordinates can be negative on multi-monitor
    # setups. MSS also accepts these coordinates for screen capture.
    left: int
    top: int
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height


class Observation(BaseModel):
    timestamp: float
    frame_id: int
    state: AutomationState
    data: dict[str, Any] = Field(default_factory=dict)


class Action(BaseModel):
    name: str
    payload: dict[str, Any] = Field(default_factory=dict)


class AutomationConfig(BaseModel):
    capture_interval_seconds: float = Field(default=0.20, gt=0)
    emergency_stop_enabled: bool = True
