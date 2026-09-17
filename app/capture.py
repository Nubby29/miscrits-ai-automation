"""Desktop capture abstraction using MSS and native Windows window capture."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any

from mss import mss
from PIL import Image

from .core.models import Region


@dataclass
class Frame:
    frame_id: int
    timestamp: float
    width: int
    height: int
    pixels: Any
    image: Image.Image | None = None

    def to_image(self) -> Image.Image:
        if self.image is not None:
            return self.image.copy()
        return Image.frombytes("RGB", (self.width, self.height), self.pixels.rgb)


class ScreenCapture:
    def __init__(self) -> None:
        self._mss = mss()
        self._frame_id = 0

    def _next_id(self) -> int:
        self._frame_id += 1
        return self._frame_id

    def grab(self, region: Region) -> Frame:
        monitor = {
            "left": region.left,
            "top": region.top,
            "width": region.width,
            "height": region.height,
        }
        image = self._mss.grab(monitor)
        return Frame(self._next_id(), time.time(), image.width, image.height, image)

    def grab_window(self, hwnd: int) -> Frame:
        """Capture a native window without depending on what is covering it."""
        if not isinstance(hwnd, int) or hwnd <= 0:
            raise ValueError("A valid Windows HWND is required")

        user32 = ctypes.windll.user32
        gdi32 = ctypes.windll.gdi32
        hwnd_value = ctypes.c_void_p(hwnd)

        # Do not force HWND through wintypes.HWND. On some Python/Windows
        # combinations that conversion can overflow even for a valid 64-bit HWND.
        user32.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.RECT)]
        user32.GetWindowRect.restype = wintypes.BOOL
        user32.GetWindowDC.argtypes = [ctypes.c_void_p]
        user32.GetWindowDC.restype = ctypes.c_void_p
        user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        user32.ReleaseDC.restype = ctypes.c_int
        user32.PrintWindow.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT]
        user32.PrintWindow.restype = wintypes.BOOL

        rect = wintypes.RECT()
        if not user32.GetWindowRect(hwnd_value, ctypes.byref(rect)):
            raise OSError("GetWindowRect failed")
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            raise OSError("Target window has invalid dimensions")

        hwnd_dc = user32.GetWindowDC(hwnd_value)
        if not hwnd_dc:
            raise OSError("GetWindowDC failed")
        gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
        gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
        gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
        gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        gdi32.SelectObject.restype = ctypes.c_void_p
        gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
        gdi32.DeleteObject.restype = wintypes.BOOL
        gdi32.DeleteDC.argtypes = [ctypes.c_void_p]
        gdi32.DeleteDC.restype = wintypes.BOOL

        mem_dc = gdi32.CreateCompatibleDC(hwnd_dc)
        bitmap = gdi32.CreateCompatibleBitmap(hwnd_dc, width, height)
        if not mem_dc or not bitmap:
            if mem_dc:
                gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd_value, hwnd_dc)
            raise OSError("Unable to create compatible bitmap")

        old_bitmap = gdi32.SelectObject(mem_dc, bitmap)
        try:
            # PW_RENDERFULLCONTENT is supported by modern Windows for many
            # applications and gives the renderer a chance to paint its content.
            result = user32.PrintWindow(hwnd_value, mem_dc, 0x00000002)
            if not result:
                result = user32.PrintWindow(hwnd_value, mem_dc, 0)
            if not result:
                raise OSError("PrintWindow failed for target window")

            class BitmapInfoHeader(ctypes.Structure):
                _fields_ = [
                    ("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG),
                    ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                    ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD),
                    ("biSizeImage", wintypes.DWORD), ("biXPelsPerMeter", wintypes.LONG),
                    ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                    ("biClrImportant", wintypes.DWORD),
                ]

            class BitmapInfo(ctypes.Structure):
                _fields_ = [("bmiHeader", BitmapInfoHeader), ("bmiColors", wintypes.DWORD * 3)]

            info = BitmapInfo()
            info.bmiHeader.biSize = ctypes.sizeof(BitmapInfoHeader)
            info.bmiHeader.biWidth = width
            info.bmiHeader.biHeight = -height
            info.bmiHeader.biPlanes = 1
            info.bmiHeader.biBitCount = 32
            info.bmiHeader.biCompression = 0

            buffer = (ctypes.c_ubyte * (width * height * 4))()
            gdi32.GetDIBits.argtypes = [ctypes.c_void_p, ctypes.c_void_p, wintypes.UINT, wintypes.UINT, ctypes.c_void_p, ctypes.POINTER(BitmapInfo), wintypes.UINT]
            gdi32.GetDIBits.restype = ctypes.c_int
            copied = gdi32.GetDIBits(mem_dc, bitmap, 0, height, ctypes.byref(buffer), ctypes.byref(info), 0)
            if copied != height:
                raise OSError("GetDIBits failed")
            image = Image.frombuffer("RGBA", (width, height), buffer, "raw", "BGRA", 0, 1).convert("RGB")
        finally:
            gdi32.SelectObject(mem_dc, old_bitmap)
            gdi32.DeleteObject(bitmap)
            gdi32.DeleteDC(mem_dc)
            user32.ReleaseDC(hwnd_value, hwnd_dc)

        return Frame(self._next_id(), time.time(), width, height, None, image)

    def close(self) -> None:
        self._mss.close()
