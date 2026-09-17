# Miscrits AI Automation

A computer-vision-driven PvE automation research project for **Miscrits: World of Creatures**.

The project observes the game through its visible Windows UI, classifies the current screen, reads battle HUD information with OCR, and can perform controlled mouse clicks. It does not use game-memory modification or anti-cheat bypass techniques.

## Current milestone

**v0.4.0 — PvE Battle Controller**

- Native game-window capture
- Exploration/battle screen classification
- Battle HUD OCR
- HP, turn, capture-percentage and ability-slot recognition
- Multi-frame observation stability
- Controlled visible-UI mouse input
- Conservative PvE battle controller
- Emergency-stop gate remains in the input layer

## PvE controller

First test the controller without clicking:

```powershell
python -m app.pve_controller --dry-run
```

When the logs correctly identify the player's turn and an ability slot, run live visible-UI PvE battle control:

```powershell
python -m app.pve_controller
```

The controller currently:

1. Finds the `Miscrits` window.
2. Captures the window directly.
3. Waits for a stable battle observation.
4. Waits for `It's your turn`.
5. Selects a recognized non-utility ability.
6. Clicks that ability through the normal desktop UI.
7. Waits for the next state before acting again.
8. Does nothing when the observation is ambiguous.

Stop with `Ctrl+C`.

## Architecture

```text
Screen Capture
      ↓
Vision / OCR
      ↓
Stable Game State
      ↓
PvE Decision Engine
      ↓
Controlled Input
      ↓
Visible Game UI
```

## Planned PvE milestones

1. Core automation framework ✅
2. Desktop capture and window selection ✅
3. UI/OCR detectors ✅
4. Battle-state recognition ✅
5. Deterministic battle controller ✅
6. Capture/collection policy
7. Healing and party switching
8. Exploration/encounter loop
9. Leveling loop
10. Long-running recovery and run history

## Scope

This project is limited to visible UI automation for PvE/testing workflows. It does not include memory injection, credential theft, anti-cheat bypassing, or techniques intended to evade game security systems.

Before using automation with a live online game, check that the behavior is permitted by the game's rules and platform policies.

## Development

Python is the initial implementation language.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the observation dashboard with:

```powershell
python -m app.desktop_dashboard
```
