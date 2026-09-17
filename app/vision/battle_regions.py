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

# Text-only portions of the four standard action buttons. Each crop is kept
# close to the visible label so neighboring button text cannot contaminate the
# slot result. Slot order is preserved for later state/strategy work.
BATTLE_ABILITY_REGIONS = (
    NormalizedRegion("ability_1", 0.285, 0.870, 0.075, 0.060),
    NormalizedRegion("ability_2", 0.415, 0.870, 0.085, 0.060),
    NormalizedRegion("ability_3", 0.550, 0.870, 0.065, 0.060),
    NormalizedRegion("ability_4", 0.670, 0.870, 0.085, 0.060),
)
