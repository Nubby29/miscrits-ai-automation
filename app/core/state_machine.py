"""Small, explicit state machine for automation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from .models import AutomationState


@dataclass
class StateMachine:
    state: AutomationState = AutomationState.IDLE
    history: list[AutomationState] = field(default_factory=list)
    handlers: dict[AutomationState, Callable[[], None]] = field(default_factory=dict)

    def transition(self, new_state: AutomationState) -> None:
        if new_state == self.state:
            return
        self.history.append(self.state)
        self.state = new_state

    def register(self, state: AutomationState, handler: Callable[[], None]) -> None:
        self.handlers[state] = handler

    def tick(self) -> None:
        handler = self.handlers.get(self.state)
        if handler is not None:
            handler()
