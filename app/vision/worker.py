"""Background vision worker that keeps expensive OCR off the Tkinter UI thread."""

from __future__ import annotations

import threading
import time
from typing import Callable

from ..capture import Frame
from .engine import VisionEngine
from .models import VisionResult


class VisionWorker:
    """Analyze only the newest submitted frame at a controlled rate.

    The worker is observation-only: it never sends input to the game. Frames are
    intentionally coalesced so a slow OCR pass cannot create an unbounded queue.
    """

    def __init__(
        self,
        engine: VisionEngine,
        on_result: Callable[[VisionResult], None],
        on_error: Callable[[Exception], None] | None = None,
        interval_seconds: float = 0.5,
    ) -> None:
        self.engine = engine
        self.on_result = on_result
        self.on_error = on_error
        self.interval_seconds = max(0.1, interval_seconds)
        self._condition = threading.Condition()
        self._latest_frame: Frame | None = None
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._condition:
            if self._running:
                return
            self._running = True
            self._thread = threading.Thread(target=self._run, name="vision-worker", daemon=True)
            self._thread.start()

    def submit(self, frame: Frame) -> None:
        with self._condition:
            if not self._running:
                return
            self._latest_frame = frame
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._running = False
            self._latest_frame = None
            self._condition.notify_all()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        self._thread = None

    def _run(self) -> None:
        next_allowed = 0.0
        while True:
            with self._condition:
                while self._running and self._latest_frame is None:
                    self._condition.wait()
                if not self._running:
                    return
                now = time.monotonic()
                wait_for = next_allowed - now
                if wait_for > 0:
                    self._condition.wait(timeout=wait_for)
                    continue
                frame = self._latest_frame
                self._latest_frame = None

            if frame is None:
                continue

            started = time.perf_counter()
            try:
                result = self.engine.analyze(frame)
                self.on_result(result)
            except Exception as exc:
                if self.on_error:
                    self.on_error(exc)
            finally:
                elapsed = time.perf_counter() - started
                next_allowed = time.monotonic() + max(0.0, self.interval_seconds - elapsed)
