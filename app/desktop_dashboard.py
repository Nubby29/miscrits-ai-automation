"""Desktop dashboard for observing a selected game window and its vision result."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageTk

from .capture import ScreenCapture
from .vision.engine import VisionEngine
from .window_manager import GameWindow, activate_window, list_windows


class DesktopDashboard:
    REFRESH_MS = 250
    SELF_TITLE = "Miscrits AI Automation — Vision Console"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(self.SELF_TITLE)
        self.root.geometry("1100x760")
        self.root.minsize(900, 650)
        self.capture = ScreenCapture()
        self.vision = VisionEngine()
        self.windows: list[GameWindow] = []
        self.selected: GameWindow | None = None
        self.running = False
        self.capture_mode = tk.StringVar(value="Window content")
        self.title_var = tk.StringVar(value="No window selected")
        self.status_var = tk.StringVar(value="Idle — observation only")
        self.fps_var = tk.StringVar(value="Capture: —")
        self.vision_var = tk.StringVar(value="Vision: —")
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
        self.window_combo = ttk.Combobox(controls, state="readonly", width=55)
        self.window_combo.pack(side="left", padx=8)
        self.window_combo.bind("<<ComboboxSelected>>", self._select_window)
        ttk.Button(controls, text="Refresh", command=self.refresh_windows).pack(side="left", padx=4)
        ttk.Button(controls, text="Activate", command=self.activate_selected).pack(side="left", padx=4)
        ttk.Label(controls, text="Capture:").pack(side="left", padx=(12, 4))
        mode_combo = ttk.Combobox(controls, textvariable=self.capture_mode, state="readonly", width=18, values=("Window content", "Screen region"))
        mode_combo.pack(side="left")
        self.toggle_button = ttk.Button(controls, text="Start Capture", command=self.toggle_capture)
        self.toggle_button.pack(side="right")
        info = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.title_var).pack(side="left")
        ttk.Label(info, textvariable=self.fps_var).pack(side="right")
        self.preview = ttk.Label(self.root, anchor="center", relief="sunken")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        vision_bar = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        vision_bar.pack(fill="x")
        ttk.Label(vision_bar, textvariable=self.vision_var).pack(side="left")

    def refresh_windows(self) -> None:
        """Refresh windows while excluding this dashboard from capture targets."""
        previous_title = self.selected.title if self.selected else None
        all_windows = list_windows()
        self.windows = [w for w in all_windows if w.title != self.SELF_TITLE]
        labels = [f"{w.title}  [{w.region.width}×{w.region.height}]" for w in self.windows]
        self.window_combo["values"] = labels
        if labels:
            index = next((i for i, w in enumerate(self.windows) if w.title == previous_title), 0)
            self.window_combo.current(index)
            self._select_window()
            self.status_var.set(f"Found {len(labels)} capture targets")
        else:
            self.window_combo.set("")
            self.selected = None
            self.title_var.set("No capture target found")
            self.status_var.set("No capture targets detected")

    def _select_window(self, _event: object | None = None) -> None:
        index = self.window_combo.current()
        if 0 <= index < len(self.windows):
            self.selected = self.windows[index]
            self.title_var.set(self.selected.title)

    def activate_selected(self) -> None:
        if not self.selected:
            return
        # Starting first keeps the observation loop alive while the target is
        # brought forward. Native window capture then remains valid if the
        # dashboard is later placed over the target.
        if not self.running:
            self.running = True
            self.toggle_button.configure(text="Stop Capture")
            self.status_var.set("Capturing + vision — no game input is being sent")
            self._capture_loop()
        if activate_window(self.selected):
            self.status_var.set(f"Active: {self.selected.title} • capture running")
        else:
            self.status_var.set("Could not activate selected window")

    def toggle_capture(self) -> None:
        self.running = not self.running
        self.toggle_button.configure(text="Stop Capture" if self.running else "Start Capture")
        self.status_var.set("Capturing + vision — no game input is being sent" if self.running else "Idle — observation only")
        if self.running:
            self._capture_loop()

    def _capture_loop(self) -> None:
        if not self.running:
            return
        if self.selected:
            started = time.perf_counter()
            try:
                if self.capture_mode.get() == "Window content" and self.selected.hwnd:
                    frame = self.capture.grab_window(self.selected.hwnd)
                else:
                    frame = self.capture.grab(self.selected.region)
                image = frame.to_image()
                self._show_image(image)
                result = self.vision.analyze(frame)
                elapsed = time.perf_counter() - started
                fps = 1 / elapsed if elapsed > 0 else 0
                self.fps_var.set(f"Capture: {fps:.1f} FPS  •  {frame.width}×{frame.height}")
                self.vision_var.set(f"Vision: {result.screen_type}  •  OCR: {len(result.ocr_text)} text items  •  Regions: {len(result.regions)}")
            except Exception as exc:
                self.status_var.set(f"Vision/capture error: {exc}")
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
