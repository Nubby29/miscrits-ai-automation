"""Safe input abstraction.

The implementation deliberately separates input commands from the decision engine.
A kill switch can stop all future input before a command is dispatched.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InputController:
    enabled: bool = True
    emergency_stopped: bool = False

    def emergency_stop(self) -> None:
        self.emergency_stopped = True

    def reset_emergency_stop(self) -> None:
        self.emergency_stopped = False

    def can_dispatch(self) -> bool:
        return self.enabled and not self.emergency_stopped

    def dispatch(self, command: str, **payload: object) -> bool:
        """Accept a command only when the controller is live.

        Actual OS input is intentionally added in a later milestone after the
        observation pipeline is tested against a controlled environment.
        """
        if not self.can_dispatch():
            return False
        # Placeholder: record/route the command to a concrete input backend later.
        _ = command, payload
        return True
