"""PvE battle automation using visible window capture and normal UI clicks.

This module intentionally avoids game-memory access. It waits for repeated
battle observations, waits for the player's turn, then clicks a recognized
ability button. Unknown/ambiguous states are left untouched.
"""

from __future__ import annotations

import argparse
import logging
import time

from .capture import ScreenCapture
from .core.input_controller import InputController
from .vision.battle_regions import BATTLE_ABILITY_REGIONS
from .vision.engine import VisionEngine
from .vision.temporal import StabilityConfig, VisionStabilityTracker
from .window_manager import activate_window, find_window

LOG = logging.getLogger(__name__)


class PVEController:
    """Conservative PvE battle loop driven entirely by visible UI state."""

    def __init__(
        self,
        *,
        input_controller: InputController | None = None,
        scan_interval: float = 0.35,
        action_cooldown: float = 1.25,
        stable_frames: int = 3,
    ) -> None:
        self.input = input_controller or InputController()
        self.scan_interval = max(0.15, scan_interval)
        self.action_cooldown = max(0.75, action_cooldown)
        self.vision = VisionEngine()
        self.tracker = VisionStabilityTracker(StabilityConfig(required_frames=max(3, stable_frames)))
        self._last_action = 0.0
        self._last_signature: tuple[object, ...] | None = None

    @staticmethod
    def _slot_point(slot: int) -> tuple[float, float]:
        region = BATTLE_ABILITY_REGIONS[slot]
        return region.x + region.width / 2, region.y + region.height / 2

    @staticmethod
    def _choose_slot(ability_slots: tuple[str | None, ...]) -> int | None:
        """Prefer a recognized offensive action and avoid known buff/flee labels."""
        preferred: list[tuple[int, str]] = []
        for index, name in enumerate(ability_slots):
            if name:
                preferred.append((index, name.casefold()))
        if not preferred:
            return None

        for index, name in preferred:
            if name not in {"power up", "flee", "switch", "capture", "item"}:
                return index
        return preferred[0][0]

    def run(self, title_contains: str = "Miscrits") -> None:
        window = find_window(title_contains)
        if window is None or window.hwnd is None:
            raise RuntimeError(f"Could not find a usable window containing: {title_contains!r}")
        if not activate_window(window):
            raise RuntimeError(f"Could not activate target window: {window.title!r}")

        LOG.info("PvE controller attached to %s", window.title)
        LOG.info("Only visible UI clicks are enabled; Ctrl+C stops the loop")
        capture = ScreenCapture()
        try:
            while self.input.can_dispatch():
                # Refresh the bounds so clicks follow a moved/resized game window.
                refreshed = find_window(title_contains)
                if refreshed is not None and refreshed.hwnd == window.hwnd:
                    window = refreshed

                frame = capture.grab_window(window.hwnd)
                result = self.vision.analyze(frame)
                stable = self.tracker.update(result)

                if stable is None:
                    time.sleep(self.scan_interval)
                    continue

                battle = stable.observation
                signature = (
                    battle.player_name,
                    battle.enemy_name,
                    battle.player_hp_current,
                    battle.enemy_hp_current,
                    battle.turn,
                    battle.capture_percent,
                    battle.ability_slots,
                )

                if result.screen_type != "battle" or battle.turn != "player":
                    time.sleep(self.scan_interval)
                    continue
                if battle.player_hp_current is not None and battle.player_hp_current <= 0:
                    LOG.info("Player HP is zero; waiting for the game to transition")
                    time.sleep(self.scan_interval)
                    continue
                if time.monotonic() - self._last_action < self.action_cooldown:
                    time.sleep(self.scan_interval)
                    continue
                if signature == self._last_signature:
                    time.sleep(self.scan_interval)
                    continue

                slot = self._choose_slot(battle.ability_slots)
                if slot is None:
                    LOG.info("No reliable ability label yet; no click")
                    time.sleep(self.scan_interval)
                    continue

                x, y = self._slot_point(slot)
                LOG.info(
                    "PvE action: slot=%d ability=%s enemy=%s HP=%s/%s",
                    slot + 1,
                    battle.ability_slots[slot],
                    battle.enemy_name or "?",
                    battle.enemy_hp_current if battle.enemy_hp_current is not None else "?",
                    battle.enemy_hp_max if battle.enemy_hp_max is not None else "?",
                )
                if self.input.dispatch("click", window_region=window.region, x=x, y=y):
                    self._last_action = time.monotonic()
                    self._last_signature = signature
                else:
                    LOG.warning("Input dispatch was rejected")
                time.sleep(self.scan_interval)
        except KeyboardInterrupt:
            LOG.info("Stopped by user")
        finally:
            capture.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visible-UI PvE battle automation for Miscrits")
    parser.add_argument("--window", default="Miscrits", help="Substring of the game window title")
    parser.add_argument("--dry-run", action="store_true", help="Observe and log actions without clicking")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    controller = InputController(enabled=not args.dry_run)
    PVEController(input_controller=controller).run(args.window)


if __name__ == "__main__":
    main()
