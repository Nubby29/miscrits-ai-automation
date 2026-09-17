"""Controlled desktop input for the PvE automation layer.

All input remains behind an explicit controller and emergency-stop gate. The
controller only targets the selected game window's client area and exposes
simple mouse/keyboard primitives for visible UI interaction.
"""

from __future__ import annotations

from dataclasses import dataclass
import time

try:
    from pynput import keyboard, mouse
except ImportError:  # pragma: no cover - dependency is installed at runtime.
    keyboard = None
    mouse = None

from .models import Region


@dataclass
class InputController:
    enabled: bool = True
    emergency_stopped: bool = False
    click_pause_seconds: float = 0.12

    def emergency_stop(self) -> None:
        self.emergency_stopped = True

    def reset_emergency_stop(self) -> None:
        self.emergency_stopped = False

    def can_dispatch(self) -> bool:
        return self.enabled and not self.emergency_stopped

    def dispatch(self, command: str, **payload: object) -> bool:
        """Dispatch a supported visible-UI command when the controller is live."""
        if not self.can_dispatch():
            return False
        if mouse is None or keyboard is None:
            return False

        if command == "click":
            region = payload.get("window_region")
            x = payload.get("x")
            y = payload.get("y")
            if not isinstance(region, Region) or not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
                return False
            if not 0.0 <= float(x) <= 1.0 or not 0.0 <= float(y) <= 1.0:
                return False
            px = round(region.left + region.width * float(x))
            py = round(region.top + region.height * float(y))
            controller = mouse.Controller()
            controller.position = (px, py)
            controller.click(mouse.Button.left, 1)
            time.sleep(max(0.0, self.click_pause_seconds))
            return True

        if command == "key":
            key = payload.get("key")
            if not isinstance(key, str) or len(key) != 1:
                return False
            controller = keyboard.Controller()
            controller.press(key)
            controller.release(key)
            time.sleep(max(0.0, self.click_pause_seconds))
            return True

        return False
