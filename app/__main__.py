"""CLI smoke test for the project foundation."""

from __future__ import annotations

import logging

from .core.input_controller import InputController
from .core.models import AutomationState, Region
from .core.state_machine import StateMachine


logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    machine = StateMachine()
    input_controller = InputController()

    machine.register(AutomationState.IDLE, lambda: machine.transition(AutomationState.STARTING))
    machine.register(AutomationState.STARTING, lambda: machine.transition(AutomationState.OBSERVING))
    machine.register(
        AutomationState.OBSERVING,
        lambda: log.info("Observation cycle ready; capture region=%s", Region(left=0, top=0, width=1280, height=720)),
    )

    log.info("Miscrits AI Automation v0.1.0")
    log.info("Initial state: %s", machine.state.value)
    machine.tick()
    machine.tick()
    machine.tick()
    log.info("Current state: %s", machine.state.value)

    input_controller.emergency_stop()
    log.info("Emergency stop engaged: %s", not input_controller.can_dispatch())


if __name__ == "__main__":
    main()
