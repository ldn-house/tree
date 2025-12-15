# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Christmas tree LED controller for 500 WS2811 12V LEDs on a 6ft tree. Inspired by Matt Parker's 3D Christmas tree project. Goal is volumetric 3D animations once coordinate mapping is implemented.

## Hardware

- **Target**: Pi Pico 2W running MicroPython (uses PIO for precise LED timing)
- **LEDs**: 500x WS2811 12V RGB strip
- **Data signal**: 3.3V GPIO (GP0) → level shifter → 5V to LEDs

## Commands

```bash
# Flash to Pico (saves as main.py, auto-runs on boot)
./flash.py

# Run without saving to flash
./flash.py --run
```

## Architecture

- `main.py` - MicroPython LED controller with animations (runs on Pico)
- `flash.py` - uv script (PEP 723) to flash/run code on Pico via mpremote

## Adding Animations

Add new animation functions to `main.py` and register them in the `ANIMATIONS` list. Each animation should:
- Use `scale(color)` to apply brightness
- Call `strip.write()` after setting pixels
- Accept duration/cycle parameters to control length
