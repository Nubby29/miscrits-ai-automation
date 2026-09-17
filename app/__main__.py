"""Command-line entry point."""

from __future__ import annotations

import argparse
import logging

from .core.input_controller import InputController
from .core.models import AutomationState, Region
from .core.state_machine import StateMachine

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def smoke_test() -> None:
    machine = StateMachine()
    input_controller = InputController()
    machine.register(AutomationState.IDLE, lambda: machine.transition(AutomationState.STARTING))
    machine.register(AutomationState.STARTING, lambda: machine.transition(AutomationState.OBSERVING))
    machine.register(AutomationState.OBSERVING, lambda: log.info("Observation cycle ready; region=%s", Region(left=0, top=0, width=1280, height=720)))
    log.info("Miscrits AI Automation v0.3.0")
    machine.tick(); machine.tick(); machine.tick()
    log.info("Current state: %s", machine.state.value)
    input_controller.emergency_stop()
    log.info("Emergency stop engaged: %s", not input_controller.can_dispatch())


def main() -> None:
    parser = argparse.ArgumentParser(description="Miscrits AI Automation")
    parser.add_argument("--dashboard", action="store_true", help="Open the desktop observation dashboard")
    args = parser.parse_args()
    if args.dashboard:
        from .desktop_dashboard import run_dashboard
        run_dashboard()
    else:
        smoke_test()


if __name__ == "__main__":
    main()
