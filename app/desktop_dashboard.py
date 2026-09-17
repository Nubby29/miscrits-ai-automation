"""Desktop dashboard for observing a selected game window and its vision result."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageDraw, ImageTk

from .capture import Frame, ScreenCapture
from .vision.engine import VisionEngine
from .vision.fixtures import save_frame
from .vision.hp_analysis import estimate_battle_hp_bars
from .vision.models import VisionResult
from .vision.temporal import VisionStabilityTracker
from .vision.worker import VisionWorker
from .window_manager import GameWindow, activate_window, list_windows


class DesktopDashboard:
    CAPTURE_MS = 80
    VISION_INTERVAL_SECONDS = 0.5
    SELF_TITLE = "Miscrits AI Automation — Vision Console"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title(self.SELF_TITLE)
        self.root.geometry("1100x860")
        self.root.minsize(900, 740)
        self.capture = ScreenCapture()
        self.vision = VisionEngine()
        self.stability_tracker = VisionStabilityTracker()
        self.vision_worker = VisionWorker(
            self.vision,
            on_result=self._queue_vision_result,
            on_error=self._queue_vision_error,
            interval_seconds=self.VISION_INTERVAL_SECONDS,
        )
        self.windows: list[GameWindow] = []
        self.selected: GameWindow | None = None
        self.running = False
        self.latest_frame: Frame | None = None
        self.latest_result: VisionResult | None = None
        self.pending_result: VisionResult | None = None
        self.pending_error: str | None = None
        self.capture_count = 0
        self.capture_window_started = time.perf_counter()
        self.capture_fps = 0.0
        self.capture_mode = tk.StringVar(value="Window content")
        self.show_regions = tk.BooleanVar(value=True)
        self.fixture_category = tk.StringVar(value="unknown")
        self.title_var = tk.StringVar(value="No window selected")
        self.status_var = tk.StringVar(value="Idle — observation only")
        self.fps_var = tk.StringVar(value="Capture: —")
        self.vision_var = tk.StringVar(value="Vision: —")
        self.ocr_var = tk.StringVar(value="OCR: —")
        self.battle_var = tk.StringVar(value="Battle: —")
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
        self.window_combo = ttk.Combobox(controls, state="readonly", width=42)
        self.window_combo.pack(side="left", padx=8)
        self.window_combo.bind("<<ComboboxSelected>>", self._select_window)
        ttk.Button(controls, text="Refresh", command=self.refresh_windows).pack(side="left", padx=4)
        ttk.Button(controls, text="Activate", command=self.activate_selected).pack(side="left", padx=4)
        ttk.Label(controls, text="Capture:").pack(side="left", padx=(12, 4))
        ttk.Combobox(controls, textvariable=self.capture_mode, state="readonly", width=16, values=("Window content", "Screen region")).pack(side="left")
        self.toggle_button = ttk.Button(controls, text="Start Capture", command=self.toggle_capture)
        self.toggle_button.pack(side="right")

        info = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.title_var).pack(side="left")
        ttk.Label(info, textvariable=self.fps_var).pack(side="right")

        self.preview = ttk.Label(self.root, anchor="center", relief="sunken")
        self.preview.pack(fill="both", expand=True, padx=12, pady=(0, 8))

        vision_bar = ttk.Frame(self.root, padding=(12, 0, 12, 4))
        vision_bar.pack(fill="x")
        ttk.Label(vision_bar, textvariable=self.vision_var).pack(side="left")
        ttk.Label(vision_bar, textvariable=self.ocr_var).pack(side="right")

        battle_bar = ttk.Frame(self.root, padding=(12, 2, 12, 4))
        battle_bar.pack(fill="x")
        ttk.Label(battle_bar, textvariable=self.battle_var).pack(side="left")

        tools = ttk.Frame(self.root, padding=(12, 4, 12, 12))
        tools.pack(fill="x")
        ttk.Checkbutton(tools, text="Show detected regions", variable=self.show_regions).pack(side="left")
        ttk.Label(tools, text="Fixture:").pack(side="left", padx=(18, 4))
        ttk.Combobox(tools, textvariable=self.fixture_category, state="readonly", width=14, values=("unknown", "exploration", "battle", "menu", "inventory", "loading")).pack(side="left")
        ttk.Button(tools, text="Save Current Frame", command=self.save_current_frame).pack(side="left", padx=8)

    def refresh_windows(self) -> None:
        previous_hwnd = self.selected.hwnd if self.selected else None
        self.windows = [w for w in list_windows() if w.title != self.SELF_TITLE]
        labels = [f"{w.title}  [{w.region.width}×{w.region.height}]" for w in self.windows]
        self.window_combo["values"] = labels
        if labels:
            index = next((i for i, w in enumerate(self.windows) if w.hwnd == previous_hwnd), 0)
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
        if not self.running:
            self.running = True
            self.toggle_button.configure(text="Stop Capture")
            self.vision_worker.start()
            self._capture_loop()
        if activate_window(self.selected):
            self.status_var.set(f"Active: {self.selected.title} • capture running")
        else:
            self.status_var.set("Could not activate selected window")

    def toggle_capture(self) -> None:
        self.running = not self.running
        self.toggle_button.configure(text="Stop Capture" if self.running else "Start Capture")
        if self.running:
            self.status_var.set("Capturing + vision — no game input is being sent")
            self.vision_worker.start()
            self._capture_loop()
        else:
            self.status_var.set("Idle — observation only")
            self.vision_worker.stop()
            self.stability_tracker.reset()

    def _capture_loop(self) -> None:
        if not self.running:
            return
        self._apply_pending_vision()
        if self.selected:
            try:
                if self.capture_mode.get() == "Window content" and self.selected.hwnd:
                    frame = self.capture.grab_window(self.selected.hwnd)
                else:
                    frame = self.capture.grab(self.selected.region)
                self.latest_frame = frame
                self.capture_count += 1
                now = time.perf_counter()
                elapsed = now - self.capture_window_started
                if elapsed >= 1.0:
                    self.capture_fps = self.capture_count / elapsed
                    self.capture_count = 0
                    self.capture_window_started = now
                self.fps_var.set(f"Capture: {self.capture_fps:.1f} FPS  •  {frame.width}×{frame.height}")
                self.vision_worker.submit(frame)
                self._show_image(frame.to_image(), self.latest_result)
            except Exception as exc:
                self.status_var.set(f"Capture error: {exc}")
        self.root.after(self.CAPTURE_MS, self._capture_loop)

    def _queue_vision_result(self, result: VisionResult) -> None:
        self.pending_result = result

    def _queue_vision_error(self, exc: Exception) -> None:
        self.pending_error = str(exc)

    def _apply_pending_vision(self) -> None:
        if self.pending_error:
            self.status_var.set(f"Vision error: {self.pending_error}")
            self.pending_error = None
        if self.pending_result is None:
            return
        result = self.pending_result
        self.pending_result = None
        self.latest_result = result
        self.vision_var.set(f"Vision: {result.screen_type} ({result.screen_confidence:.0%})  •  Regions: {len(result.regions)}")
        self.ocr_var.set(f"OCR: {len(result.ocr_text)} text items")
        stable = self.stability_tracker.update(result)
        self._update_battle(result, stable)

    def _update_battle(self, result: VisionResult, stable=None) -> None:
        battle = result.battle
        if not battle:
            self.battle_var.set("Battle: —")
            return
        details = []
        if battle.player_name:
            details.append(f"Player: {battle.player_name}")
        if battle.enemy_name:
            details.append(f"Enemy: {battle.enemy_name}")
        if battle.player_hp_current is not None and battle.player_hp_max is not None:
            details.append(f"Player HP: {battle.player_hp_current}/{battle.player_hp_max}")
        if battle.enemy_hp_current is not None and battle.enemy_hp_max is not None:
            details.append(f"Enemy HP: {battle.enemy_hp_current}/{battle.enemy_hp_max}")
        if battle.turn:
            details.append(f"Turn: {battle.turn}")
        if battle.capture_percent is not None:
            details.append(f"Capture: {battle.capture_percent}%")
        if battle.ability_slots:
            slots = " | ".join(f"{i + 1}:{name or '—'}" for i, name in enumerate(battle.ability_slots))
            details.append("Abilities: " + slots)
        elif battle.abilities:
            details.append("Abilities: " + ", ".join(battle.abilities))

        # Visual HP analysis is diagnostic only. Numeric OCR remains the
        # authoritative HP value until fixture measurements prove otherwise.
        if self.latest_frame is not None:
            try:
                bars = estimate_battle_hp_bars(self.latest_frame.to_image())
                visual_parts = []
                for side in ("player", "enemy"):
                    estimate = bars[side]
                    if estimate.percent is not None:
                        visual_parts.append(f"{side} bar {estimate.percent}% ({estimate.confidence:.0%})")
                if visual_parts:
                    details.append("Visual: " + ", ".join(visual_parts))
            except Exception as exc:
                result.diagnostics["hp_bar_error"] = str(exc)

        if stable is not None:
            details.append(f"Stable: {stable.stable_frames} frames ({stable.confidence:.0%})")
        else:
            details.append("Stable: waiting for repeated frames")
        self.battle_var.set("Battle: " + ("  •  ".join(details) if details else "detected"))

    def _show_image(self, image: Image.Image, result: VisionResult | None) -> None:
        display = image.copy()
        if self.show_regions.get() and result and result.regions:
            draw = ImageDraw.Draw(display)
            for region in result.regions:
                draw.rectangle((region.left, region.top, region.right, region.bottom), outline="red", width=3)
                draw.text((region.left + 4, region.top + 4), region.name, fill="red")
        width = max(self.preview.winfo_width(), 640)
        height = max(self.preview.winfo_height(), 480)
        display.thumbnail((width - 20, height - 20), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(display)
        self.preview.configure(image=photo, text="")
        self.preview.image = photo

    def save_current_frame(self) -> None:
        if self.latest_frame is None:
            self.status_var.set("No frame available — start capture first")
            return
        try:
            category = self.fixture_category.get().strip() or "unknown"
            path = save_frame(self.latest_frame, category)
            self.status_var.set(f"Fixture saved: {path}")
        except Exception as exc:
            self.status_var.set(f"Fixture save error: {exc}")

    def close(self) -> None:
        self.running = False
        self.vision_worker.stop()
        self.capture.close()
        self.root.destroy()


def run_dashboard() -> None:
    root = tk.Tk()
    DesktopDashboard(root)
    root.mainloop()
