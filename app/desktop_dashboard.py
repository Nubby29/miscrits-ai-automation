"""Small local desktop dashboard for observing the selected game window.

This is intentionally observation-first: it captures and displays the selected
window but does not automatically send game actions.
"""

from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .capture import ScreenCapture
from .core.models import Region
from .window_manager import GameWindow, activate_window, list_windows


class DesktopDashboard:
    REFRESH_MS = 250

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Miscrits AI Automation — Vision Console")
        self.root.geometry("1100x760")
        self.root.minsize(900, 650)

        self.capture = ScreenCapture()
        self.windows: list[GameWindow] = []
        self.selected: GameWindow | None = None
        self.running = False
        self.latest_image: Image.Image | None = None

        self.title_var = tk.StringVar(value="No window selected")
        self.status_var = tk.StringVar(value="Idle — observation only")
        self.fps_var = tk.StringVar(value="Capture: —")

        self._build_ui()
        self.refresh_windows()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, padding=12)
        header.pack(fill="x")
        ttk.Label(header, text="MISCRITS AI AUTOMATION", font=("Segoe UI", 16, "bold")).pack(side="left")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        controls = ttk.Frame(self.root, padding=(12, 0, 12, 10))
        controls.pack(fill="x")
        ttk.Label(controls, text="Game window:").pack(side="left")
        self.window_combo = ttk.Combobox(controls, state="readonly", width=65)
        self.window_combo.pack(side="left", padx=8)
        self.window_combo.bind("<<ComboboxSelected>>", self._select_window)
        ttk.Button(controls, text="Refresh", command=self.refresh_windows).pack(side="left", padx=4)
        ttk.Button(controls, text="Activate", command=self.activate_selected).pack(side="left", padx=4)
        self.toggle_button = ttk.Button(controls, text="Start Capture", command=self.toggle_capture)
        self.toggle_button.pack(side="right")

        info = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.title_var).pack(side="left")
        ttk.Label(info, textvariable=self.fps_var).pack(side="right")

        self.preview = ttk.Label(self.root, anchor="center", relief="sunken")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def refresh_windows(self) -> None:
        self.windows = list_windows()
        labels = [f"{w.title}  [{w.region.width}×{w.region.height}]" for w in self.windows]
        self.window_combo["values"] = labels
        if labels:
            self.window_combo.current(0)
            self._select_window()
            self.status_var.set(f"Found {len(labels)} visible windows")
        else:
            self.window_combo.set("")
            self.selected = None
            self.title_var.set("No usable window found")
            self.status_var.set("No windows detected")

    def _select_window(self, _event: object | None = None) -> None:
        index = self.window_combo.current()
        if 0 <= index < len(self.windows):
            self.selected = self.windows[index]
            self.title_var.set(self.selected.title)

    def activate_selected(self) -> None:
        if self.selected and activate_window(self.selected):
            self.status_var.set(f"Active: {self.selected.title}")
        elif self.selected:
            self.status_var.set("Could not activate selected window")

    def toggle_capture(self) -> None:
        self.running = not self.running
        self.toggle_button.configure(text="Stop Capture" if self.running else "Start Capture")
        self.status_var.set("Capturing — no game input is being sent" if self.running else "Idle — observation only")
        if self.running:
            self._capture_loop()

    def _capture_loop(self) -> None:
        if not self.running:
            return
        if self.selected:
            started = time.perf_counter()
            try:
                frame = self.capture.grab(self.selected.region)
                image = Image.frombytes("RGB", (frame.width, frame.height), frame.pixels.rgb)
                self.latest_image = image
                self._show_image(image)
                elapsed = time.perf_counter() - started
                fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_var.set(f"Capture: {fps:.1f} FPS  •  {frame.width}×{frame.height}")
            except Exception as exc:
                self.status_var.set(f"Capture error: {exc}")
        self.root.after(self.REFRESH_MS, self._capture_loop)

    def _show_image(self, image: Image.Image) -> None:
        width = max(self.preview.winfo_width(), 640)
        height = max(self.preview.winfo_height(), 480)
        preview = image.copy()
        preview.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(preview)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def close(self) -> None:
        self.running = False
        self.capture.close()
        self.root.destroy()


def run_dashboard() -> None:
    root = tk.Tk()
    DesktopDashboard(root)
    root.mainloop()
