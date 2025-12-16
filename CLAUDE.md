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

# Flash all scripts via USB (main, ota, mqtt, config)
./flash.py --all

# Flash all scripts + coordinates
./flash.py --all --coords

# Flash config only
./flash.py --config

# Run without saving to flash
./flash.py --run

# Push update via WiFi OTA (no USB needed)
./flash.py --ota --host 192.168.2.149

# Push specific file via OTA
./flash.py --ota --host 192.168.2.149 ota.py

# Fetch latest from GitHub CI and push via OTA
./flash.py --github --ota --host 192.168.2.149

# Fetch from GitHub CI and flash via USB
./flash.py --github
```

## Architecture

- `main.py` - MicroPython LED controller with animations (runs on Pico)
- `mqtt.py` - MQTT client for Home Assistant integration (auto-discovery, state sync)
- `ota.py` - WiFi OTA update server with HTTP API (runs in background thread)
- `config.py` - WiFi credentials, MQTT settings (gitignored)
- `flash.py` - uv script (PEP 723) to flash/run code via USB or OTA
- `simulator.html` - Browser-based simulator using Pyodide (runs same Python code)
- `serve.py` - Local HTTP server for the simulator

### Calibration Tools (for 3D coordinate mapping)

- `capture.py` - Camera capture script for LED position detection
- `analyze.py` - Image analysis with local contrast detection and wire constraint validation
- `triangulate.py` - 3D position calculation from multiple camera angles
- `coordinates/` - Generated 3D LED coordinates (CSV, JSON, compact formats)

## OTA API

Once connected to WiFi, the Pico exposes an HTTP API on port 8080:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Status: hostname, IP, uptime, current animation, MQTT status |
| `/update?file=X` | POST | Push new code, reboots after write |
| `/gc` | GET | Trigger garbage collection |
| `/mqtt` | GET | MQTT debug info (connection state, pending commands) |
| `/calibrate/start` | GET | Enter calibration mode (stops animations) |
| `/calibrate/stop` | GET | Exit calibration mode |
| `/calibrate/led?n=X` | GET | Light single LED X for position capture |
| `/calibrate/binary?bit=X` | GET | Light LEDs where bit X is set (for binary encoding) |
| `/calibrate/clear` | GET | Turn off all LEDs in calibration mode |

mDNS (`tree.local`) is partially implemented but may not work on all networks.

## MQTT / Home Assistant

The tree uses MQTT with Home Assistant auto-discovery. On boot, it publishes a discovery config to `homeassistant/light/xmas_tree/config` and HA automatically creates a light entity.

**Topics:**
- `xmas_tree/state` - Published state (power, brightness, effect)
- `xmas_tree/set` - Commands from HA (JSON schema)

**Features:**
- Power on/off
- Brightness (0-255, with HDR power limiting)
- Effects (auto-cycle or lock to specific effect)

**Power Limiting:** Total RGB output is capped per-frame at ~15% of theoretical max to prevent PSU brownouts. This allows the brightness slider to go 0-100% while automatically scaling down when many LEDs are lit.

## Simulator

Preview animations in the browser without hardware:

```bash
./serve.py  # Opens http://localhost:8000/simulator.html
```

Uses Pyodide to run the actual `main.py` Python code in WebAssembly. The simulator mocks `machine`, `neopixel`, and `time` modules to render to a canvas.

## Adding Animations

Add new animation functions to `main.py` and register them in the `ANIMATIONS` list as `(name, lambda)` tuples:

```python
ANIMATIONS = [
    ("auto", None),  # Special: cycles through all effects
    ("my_animation", lambda: my_animation(param1, param2)),
    ...
]
```

Each animation function should:
- Use `scale(color)` to apply brightness (also handles GRB color order)
- Call `safe_write()` instead of `strip.write()` for automatic power limiting
- Call `check_interrupt()` periodically and return early if True (for responsive HA control)
- Accept duration/cycle parameters to control length

Example:
```python
def my_animation(wait_ms=50, cycles=10):
    for _ in range(cycles):
        if check_interrupt():
            return
        for i in range(LED_COUNT):
            strip[i] = scale((255, 0, 0))
        safe_write()
        time.sleep_ms(wait_ms)
```
