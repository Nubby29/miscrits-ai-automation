"""Observation-only visual analysis helpers for Miscrits HP bars.

The numeric HP text is the authoritative value. The visual analyzer is a
secondary diagnostic that estimates the filled portion of the small horizontal
HUD bar and reports low confidence when the crop does not contain a clear bar.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from .regions import NormalizedRegion


@dataclass(frozen=True)
class HPBarEstimate:
    """A visual estimate of an HP bar's fill fraction."""

    percent: int | None
    confidence: float
    sample_pixels: int
    active_columns: int
    total_columns: int
    diagnostics: dict[str, float | int | str]


# Narrow strips calibrated against the actual colored HP bars in the native
# 1382x736 Miscrits capture. They intentionally avoid the name/icon and HP text.
PLAYER_HP_BAR_CANDIDATE = NormalizedRegion("player_hp_bar_candidate", 0.295, 0.074, 0.090, 0.020)
ENEMY_HP_BAR_CANDIDATE = NormalizedRegion("enemy_hp_bar_candidate", 0.690, 0.074, 0.065, 0.020)


def _column_activity(crop: Image.Image) -> list[float]:
    width, height = crop.size
    scores: list[float] = []
    for x in range(width):
        active = 0
        for y in range(height):
            r, g, b = crop.getpixel((x, y))
            maximum = max(r, g, b)
            minimum = min(r, g, b)
            saturation = (maximum - minimum) / max(1, maximum)
            brightness = (r + g + b) / 765.0
            if saturation >= 0.25 and brightness >= 0.16:
                active += 1
        scores.append(active / max(1, height))
    return scores


def estimate_hp_bar(image: Image.Image, region: NormalizedRegion) -> HPBarEstimate:
    """Estimate horizontal fill from a calibrated candidate strip."""
    left, top, right, bottom = region.crop_box(image)
    if right <= left or bottom <= top:
        return HPBarEstimate(None, 0.0, 0, 0, 0, {"reason": "invalid_region"})

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    width, height = crop.size
    if width < 12 or height < 3:
        return HPBarEstimate(None, 0.0, width * height, 0, width, {"reason": "small_region"})

    column_scores = _column_activity(crop)
    threshold = 0.40
    best_run = 0
    run = 0
    for score in column_scores:
        if score >= threshold:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0

    mean_score = sum(column_scores) / width
    if best_run < max(5, round(width * 0.08)):
        return HPBarEstimate(None, 0.0, width * height, best_run, width, {"reason": "no_stable_fill", "mean_column_score": round(mean_score, 3)})

    percent = round(best_run / width * 100)
    continuity = best_run / width
    confidence = min(0.95, max(0.0, 0.30 + continuity * 0.55 + mean_score * 0.15))
    if confidence < 0.55:
        percent = None

    return HPBarEstimate(
        percent=percent,
        confidence=confidence,
        sample_pixels=width * height,
        active_columns=best_run,
        total_columns=width,
        diagnostics={"mean_column_score": round(mean_score, 3), "continuity": round(continuity, 3), "candidate": region.name},
    )


def estimate_battle_hp_bars(image: Image.Image) -> dict[str, HPBarEstimate]:
    """Return diagnostic visual estimates for both battle HP candidates."""
    return {"player": estimate_hp_bar(image, PLAYER_HP_BAR_CANDIDATE), "enemy": estimate_hp_bar(image, ENEMY_HP_BAR_CANDIDATE)}
