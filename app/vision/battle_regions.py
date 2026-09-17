"""Fine-grained battle UI regions used by the observation-only OCR pipeline."""

from __future__ import annotations

from .regions import NormalizedRegion


# Coordinates are relative to the full native-captured Miscrits window.
# The native capture includes the small window title bar, so these are calibrated
# against the actual 1382x736 capture rather than a cropped game-only image.
BATTLE_PLAYER_NAME = NormalizedRegion("player_name", 0.285, 0.045, 0.090, 0.045)
BATTLE_PLAYER_HP = NormalizedRegion("player_hp", 0.345, 0.095, 0.060, 0.040)
BATTLE_ENEMY_NAME = NormalizedRegion("enemy_name", 0.615, 0.045, 0.110, 0.045)
BATTLE_ENEMY_HP = NormalizedRegion("enemy_hp", 0.745, 0.095, 0.060, 0.040)
BATTLE_CAPTURE_TEXT = NormalizedRegion("capture_text", 0.455, 0.135, 0.110, 0.075)
BATTLE_TURN_REGION = NormalizedRegion("turn_text", 0.445, 0.795, 0.190, 0.075)

# These regions intentionally include a little left-side padding because the
# OCR engine trims the first 22% internally to remove the circular ability icon.
# The post-trim crop therefore begins before the actual text label in each slot.
BATTLE_ABILITY_REGIONS = (
    NormalizedRegion("ability_1", 0.270, 0.870, 0.100, 0.060),
    NormalizedRegion("ability_2", 0.400, 0.870, 0.110, 0.060),
    NormalizedRegion("ability_3", 0.540, 0.870, 0.090, 0.060),
    NormalizedRegion("ability_4", 0.650, 0.870, 0.120, 0.060),
)
