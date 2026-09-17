"""Fine-grained battle UI regions used by the observation-only OCR pipeline."""

from __future__ import annotations

from .regions import NormalizedRegion


# Coordinates are relative to the captured Miscrits client window.
# The game HUD is stable enough that these small text crops are more reliable
# than OCR'ing the entire battle screen.
BATTLE_PLAYER_NAME = NormalizedRegion("player_name", 0.285, 0.035, 0.125, 0.040)
BATTLE_PLAYER_HP = NormalizedRegion("player_hp", 0.355, 0.067, 0.085, 0.040)
BATTLE_ENEMY_NAME = NormalizedRegion("enemy_name", 0.615, 0.035, 0.110, 0.040)
BATTLE_ENEMY_HP = NormalizedRegion("enemy_hp", 0.685, 0.067, 0.080, 0.040)
BATTLE_CAPTURE_TEXT = NormalizedRegion("capture_text", 0.445, 0.105, 0.115, 0.075)
BATTLE_TURN_REGION = NormalizedRegion("turn_text", 0.445, 0.805, 0.180, 0.060)

# Text-only portions of the four standard action buttons. The first button is
# wider in the game's layout, so its crop starts farther left.
BATTLE_ABILITY_REGIONS = (
    NormalizedRegion("ability_1", 0.270, 0.835, 0.145, 0.075),
    NormalizedRegion("ability_2", 0.405, 0.835, 0.135, 0.075),
    NormalizedRegion("ability_3", 0.535, 0.835, 0.135, 0.075),
    NormalizedRegion("ability_4", 0.665, 0.835, 0.120, 0.075),
)
