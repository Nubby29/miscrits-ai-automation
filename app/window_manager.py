"""Windows desktop-window discovery and selection helpers."""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from dataclasses import dataclass

import pygetwindow as gw

from .core.models import Region


@dataclass(frozen=True)
class GameWindow:
    title: str
    region: Region
    hwnd: int | None = None


def _native_window_rect(hwnd: int) -> tuple[int, int, int, int] | None:
    """Read reliable native bounds from Windows rather than pygetwindow metadata."""
    if not hwnd:
        return None
    try:
        user32 = ctypes.windll.user32
        rect = wintypes.RECT()
        if not user32.GetWindowRect(wintypes.HWND(hwnd), ctypes.byref(rect)):
            return None
        return int(rect.left), int(rect.top), int(rect.right), int(rect.bottom)
    except (AttributeError, OSError, ValueError):
        return None


def list_windows() -> list[GameWindow]:
    """Return visible, non-empty top-level windows with reliable bounds."""
    result: list[GameWindow] = []
    for window in gw.getAllWindows():
        title = (window.title or "").strip()
        if not title:
            continue
        raw_hwnd = getattr(window, "_hWnd", None)
        try:
            hwnd = int(raw_hwnd) if raw_hwnd is not None else None
        except (TypeError, ValueError, OverflowError):
            hwnd = None

        rect = _native_window_rect(hwnd) if hwnd else None
        if rect:
            left, top, right, bottom = rect
            width, height = right - left, bottom - top
        else:
            left, top = int(window.left), int(window.top)
            width, height = int(window.width), int(window.height)

        if width <= 40 or height <= 40:
            continue

        result.append(GameWindow(title=title, hwnd=hwnd, region=Region(left=left, top=top, width=width, height=height)))
    return result


def find_window(title_contains: str) -> GameWindow | None:
    """Find the first window whose title contains the supplied text."""
    needle = title_contains.casefold().strip()
    if not needle:
        return None
    return next((w for w in list_windows() if needle in w.title.casefold()), None)


def activate_window(window: GameWindow) -> bool:
    """Bring a selected window to the foreground using its stable HWND."""
    if window.hwnd:
        try:
            user32 = ctypes.windll.user32
            hwnd = wintypes.HWND(window.hwnd)
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)  # SW_RESTORE
            if user32.SetForegroundWindow(hwnd):
                return True
        except (AttributeError, OSError, ValueError):
            pass

    for candidate in gw.getAllWindows():
        candidate_hwnd = getattr(candidate, "_hWnd", None)
        if (window.hwnd is not None and candidate_hwnd == window.hwnd) or candidate.title == window.title:
            try:
                if candidate.isMinimized:
                    candidate.restore()
                candidate.activate()
                return True
            except Exception:
                return False
    return False
