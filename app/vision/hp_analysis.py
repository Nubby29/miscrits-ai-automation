"""Observation-only visual analysis helpers for Miscrits HP bars.

This module deliberately does not interact with the game. It estimates the
filled fraction of a horizontal HP-bar candidate using pixel color/saturation
statistics. OCR remains the authoritative numeric source until fixtures prove
that a visual bar estimate is stable enough to promote.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageStat

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


# These are intentionally broad candidate strips inside the status panels.
# They are diagnostic candidates, not a hard-coded assertion about the exact
# game asset layout. Fixture validation can narrow them later.
PLAYER_HP_BAR_CANDIDATE = NormalizedRegion("player_hp_bar_candidate", 0.225, 0.085, 0.155, 0.030)
ENEMY_HP_BAR_CANDIDATE = NormalizedRegion("enemy_hp_bar_candidate", 0.615, 0.085, 0.155, 0.030)


def estimate_hp_bar(image: Image.Image, region: NormalizedRegion) -> HPBarEstimate:
    """Estimate horizontal fill from colored/saturated pixels in a candidate strip."""
    left, top, right, bottom = region.crop_box(image)
    if right <= left or bottom <= top:
        return HPBarEstimate(None, 0.0, 0, 0, 0, {"reason": "invalid_region"})

    crop = image.crop((left, top, right, bottom)).convert("RGB")
    width, height = crop.size
    if width < 8 or height < 3:
        return HPBarEstimate(None, 0.0, width * height, 0, width, {"reason": "small_region"})

    # Downsample vertically so text/outline pixels have less influence while
    # preserving the horizontal fill boundary.
    column_scores: list[float] = []
    for x in range(width):
        score = 0.0
        for y in range(height):
            r, g, b = crop.getpixel((x, y))
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            saturation = (max_c - min_c) / max(1, max_c)
            # Green/cyan/red UI fills tend to be saturated relative to the
            # neutral panel background. Brightness keeps dark outlines out.
            brightness = (r + g + b) / 765.0
            if saturation >= 0.18 and brightness >= 0.16:
                score += 1.0
        column_scores.append(score / height)

    # A filled bar normally forms a contiguous run. Use the longest run of
    # columns whose score exceeds a modest threshold rather than raw pixel
    # counting, which is sensitive to rounded corners and text.
    threshold = 0.35
    best_run = 0
    run = 0
    for score in column_scores:
        if score >= threshold:
            run += 1
            best_run = max(best_run, run)
        else:
            run = 0

    if best_run < max(4, width // 20):
        return HPBarEstimate(None, 0.0, width * height, 0, width, {"reason": "no_stable_fill", "mean_column_score": round(sum(column_scores) / width, 3)})

    percent = round(best_run / width * 100)
    mean_score = sum(column_scores) / width
    continuity = best_run / max(1, width)
    confidence = min(0.92, max(0.0, 0.35 + continuity * 0.45 + mean_score * 0.20))
    return HPBarEstimate(
        percent=percent,
        confidence=confidence,
        sample_pixels=width * height,
        active_columns=best_run,
        total_columns=width,
        diagnostics={
            "mean_column_score": round(mean_score, 3),
            "continuity": round(continuity, 3),
            "candidate": region.name,
        },
    )


def estimate_battle_hp_bars(image: Image.Image) -> dict[str, HPBarEstimate]:
    """Return diagnostic visual estimates for both battle HP candidates."""
    return {
        "player": estimate_hp_bar(image, PLAYER_HP_BAR_CANDIDATE),
        "enemy": estimate_hp_bar(image, ENEMY_HP_BAR_CANDIDATE),
    }
