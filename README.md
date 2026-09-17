# Miscrits AI Automation

A computer-vision-driven game automation research project for **Miscrits: World of Creatures**.

The project is designed around visible screen interaction rather than game-memory modification. The initial goal is to build a reusable automation framework that can observe game state, classify UI states, make deterministic decisions, and send keyboard/mouse input.

## Current milestone

**v0.1.0 — Automation Core Foundation**

- Window/screen capture abstraction
- Region-of-interest support
- Automation state machine
- Input abstraction
- Structured event logging
- Emergency stop support
- Configuration model

The first implementation is intentionally game-agnostic so the same core can later support a controlled test game and then Miscrits-specific detectors.

## Architecture

```text
Screen Capture
      ↓
Vision / OCR
      ↓
Game State
      ↓
Decision Engine
      ↓
State Machine
      ↓
Mouse / Keyboard Input
      ↓
Game
```

## Safety / scope

This repository focuses on visible UI automation and computer-vision research. It does not include memory injection, anti-cheat bypassing, credential theft, or techniques intended to evade game security systems.

Before using automation with a live online game, check that the behavior is permitted by the game's rules and platform policies.

## Planned milestones

1. Core automation framework
2. Desktop capture and window selection
3. UI/OCR detectors
4. Battle-state recognition
5. Deterministic battle controller
6. Navigation experiments
7. Dashboard and run history
8. AI-assisted state interpretation
9. Controlled end-to-end testing

## Development

Python is the initial implementation language.

A virtual environment is recommended:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run the current smoke test with:

```powershell
python -m app
```
