"""Fine-grained battle UI regions used by the observation-only OCR pipeline."""

from __future__ import annotations

from .regions import NormalizedRegion


# These are intentionally small crops around the text-bearing parts of the
# battle HUD. Keeping them separate lets each OCR pass use an appropriate
# page-segmentation mode and character whitelist.
BATTLE_PLAYER_NAME = NormalizedRegion("player_name", 0.285, 0.035, 0.145, 0.045)
BATTLE_PLAYER_HP = NormalizedRegion("player_hp", 0.335, 0.060, 0.095, 0.045)
BATTLE_ENEMY_NAME = NormalizedRegion("enemy_name", 0.595, 0.035, 0.145, 0.045)
BATTLE_ENEMY_HP = NormalizedRegion("enemy_hp", 0.635, 0.060, 0.095, 0.045)
BATTLE_CAPTURE_TEXT = NormalizedRegion("capture_text", 0.445, 0.070, 0.115, 0.065)

# Four action buttons in the standard battle action bar. The coordinates are
# relative to the captured client window and are deliberately slightly inset
# to exclude icons and button borders from the text OCR.
BATTLE_ABILITY_REGIONS = (
    NormalizedRegion("ability_1", 0.340, 0.755, 0.100, 0.070),
    NormalizedRegion("ability_2", 0.440, 0.755, 0.100, 0.070),
    NormalizedRegion("ability_3", 0.540, 0.755, 0.100, 0.070),
    NormalizedRegion("ability_4", 0.640, 0.755, 0.100, 0.070),
)

BATTLE_TURN_REGION = NormalizedRegion("turn_text", 0.465, 0.735, 0.105, 0.045)
