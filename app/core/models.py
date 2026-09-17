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
    x: int
    y: int


class Region(BaseModel):
    left: int = Field(ge=0)
    top: int = Field(ge=0)
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
