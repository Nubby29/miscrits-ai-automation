"""Windows desktop-window discovery and selection helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pygetwindow as gw

from .core.models import Region


@dataclass(frozen=True)
class GameWindow:
    title: str
    region: Region


def list_windows() -> list[GameWindow]:
    """Return visible, non-empty top-level windows with usable bounds."""
    result: list[GameWindow] = []
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if not title or window.width <= 0 or window.height <= 0:
            continue
        result.append(
            GameWindow(
                title=title,
                region=Region(
                    left=int(window.left),
                    top=int(window.top),
                    width=int(window.width),
                    height=int(window.height),
                ),
            )
        )
    return result


def find_window(title_contains: str) -> GameWindow | None:
    """Find the first window whose title contains the supplied text."""
    needle = title_contains.casefold().strip()
    if not needle:
        return None
    return next((w for w in list_windows() if needle in w.title.casefold()), None)


def activate_window(window: GameWindow) -> bool:
    """Bring a selected window to the foreground."""
    for candidate in gw.getAllWindows():
        if candidate.title == window.title:
            try:
                if candidate.isMinimized:
                    candidate.restore()
                candidate.activate()
                return True
            except Exception:
                return False
    return False
