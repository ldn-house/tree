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
# Flash via USB (saves as main.py, auto-runs on boot)
./flash.py

# Flash all config files via USB (first-time setup)
./flash.py --config

# Run without saving to flash
./flash.py --run

# Push update via WiFi OTA (no USB needed)
./flash.py --ota --host 192.168.2.149

# Push specific file via OTA
./flash.py --ota --host 192.168.2.149 ota.py
```

## Architecture

- `main.py` - MicroPython LED controller with animations (runs on Pico)
- `ota.py` - WiFi OTA update server with HTTP API (runs in background thread)
- `config.py` - WiFi credentials and OTA settings
- `flash.py` - uv script (PEP 723) to flash/run code via USB or OTA

## OTA API

Once connected to WiFi, the Pico exposes an HTTP API on port 8080:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status: hostname, IP, uptime, current animation |
| `/update?file=X` | POST | Push new code, reboots after write |
| `/gc` | GET | Trigger garbage collection |

mDNS (`tree.local`) is partially implemented but may not work on all networks.

## Adding Animations

Add new animation functions to `main.py` and register them in the `ANIMATIONS` list as `(name, lambda)` tuples:

```python
ANIMATIONS = [
    ("my_animation", lambda: my_animation(param1, param2)),
    ...
]
```

Each animation function should:
- Use `scale(color)` to apply brightness
- Call `strip.write()` after setting pixels
- Accept duration/cycle parameters to control length
