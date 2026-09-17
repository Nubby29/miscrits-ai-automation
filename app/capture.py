"""Desktop capture abstraction using MSS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mss import mss

from .core.models import Region


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    width: int
    height: int
    pixels: Any


class ScreenCapture:
    def __init__(self) -> None:
        self._mss = mss()
        self._frame_id = 0

    def grab(self, region: Region) -> Frame:
        import time

        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        image = self._mss.grab(monitor)
        self._frame_id += 1
        return Frame(
            frame_id=self._frame_id,
            timestamp=time.time(),
            width=image.width,
            height=image.height,
            pixels=image,
        )

    def close(self) -> None:
        self._mss.close()
