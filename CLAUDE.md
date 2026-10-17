# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Christmas tree LED controller for 500 WS2811 12V LEDs on a 6ft tree. Inspired by Matt Parker's 3D Christmas tree project. Goal is volumetric 3D animations once coordinate mapping is implemented.

## Hardware

- **Target**: Pi Pico 2W running MicroPython (>= 1.25; bundles are mpy v6.3 bytecode)
- **LEDs**: 500x WS2811 12V RGB strip, RGB on the wire (`COLOR_ORDER`)
- **Data signal**: 3.3V GPIO (GP0) → level shifter → 5V to LEDs, driven by a PIO state machine fed by DMA

## Commands

```bash
# First install over USB: recovery layer + firmware + config.py (+ coordinates)
./flash.py --all

# Reinstall over USB (recovery layer + local build)
./flash.py

# Push a local build over Wi-Fi; waits until the device confirms it healthy
./flash.py --ota --host tree.local

# Push the latest GitHub Release (channel main or stable)
./flash.py --ota --host tree.local --github --channel main

# Status, rollback, coordinates
./flash.py --ota --host tree.local --status
./flash.py --ota --host tree.local --rollback
./flash.py --ota --host tree.local --coords

# Build the bundle only (dist/tree-firmware.bin, dist/manifest.json)
tools/build_firmware.py

# Offline firmware tests: MicroPython unix port, mocked hardware, local mosquitto
uv run --with pytest --with paho-mqtt --with mpy-cross==1.29.0.post2 pytest tests/firmware
```

OTA needs `HTTP_TOKEN` (config.py, `$TREE_TOKEN` or `--token`). config.py only ever goes over USB.

## Architecture

Firmware lives in `firmware/`:

- **Recovery layer** (USB only; nothing on the network can write it): `boot.py`, `main.py`, `bootmgr.py`, `safemode.py`.
  - `bootmgr.py` keeps `/boot.json` (active slot, previous slot, trial flag, boot counter) and installs update bundles into the inactive slot `/fw/a` or `/fw/b`, verifying size, SHA-256, signature, `.mpy` version and Python syntax. Switching is one atomic rename of `boot.json`.
  - A freshly installed slot is a *trial*: if the device reboots before the app reports healthy, the next boot rolls back. A confirmed slot that fails 3 boots in a row falls back to the previous slot, then to **safe mode** (`safemode.py`: Wi-Fi + minimal authenticated HTTP server that accepts a new bundle).
  - `boot.py` only keeps an already-running watchdog fed by a timer, so mpremote sessions (which soft-reset without running main.py) are never cut off. The boot counter is in main.py's path.
- **App** (`firmware/app/`, one slot, replaced by updates): a single asyncio event loop, no threads.
  - `runtime.py` wires tasks together; `supervisor.py` owns the 8 s hardware watchdog and feeds it only while every task heartbeats, decides health, confirms trial updates and performs reboots.
  - `engine.py` runs effects from the `effects/` package (docs/effect-api.md): render → output stage (`effects/_output.py`: power cap on the raw frame, brightness, gamma, colour order) → `driver.py` (PIO + DMA). Effect exceptions mark the effect failed and the scheduler moves on; an effect that hangs is quarantined for the next boot (watchdog scratch register + `/fx_probe.txt`).
  - `state.py`: power, brightness, effect, speed, auto-cycle duration and params, persisted to `/state.json` and authoritative at boot.
  - `net.py` (Wi-Fi state machine), `dns.py` (async DNS), `mqtt.py` (async MQTT 3.1.1 client with LWT, keepalive dead-peer detection, backoff), `ha.py` (Home Assistant discovery and commands).
  - `httpd.py` + `api.py`: HTTP API with token auth; `updates.py`: bundle uploads and GitHub Releases self-update; `calib.py`: calibration lock and sequence player (docs/calibration-protocol.md), with the vendored `seqfile.py` codec.
- `tools/build_firmware.py` builds the bundle (mpy-cross compiles `firmware/app` and `effects/`, adds `data/effects.json` metadata, legacy coordinates and the default calibration sequence, signs with `FIRMWARE_SIGNING_KEY`).
- `flash.py` — uv script (PEP 723) to install over USB (mpremote) or OTA.
- `tests/firmware/` — the firmware on the MicroPython unix port: `mocks/` (machine, rp2, network), `devsim.py` (simulated resets and watchdog), unit tests in `mp/`, and device tests for updates, faults, MQTT and calibration.
- `config.py` — WiFi, token, MQTT, signing key (gitignored; see `config.example.py`)

Flash layout on the device: recovery files and `config.py` at `/`, slots in `/fw/a` and `/fw/b`, hot-pushed effects in `/fx`, plus `/boot.json`, `/state.json`, `/coords.txt`.

### Calibration Tools (for 3D coordinate mapping)

- `capture.py` - Camera capture script for LED position detection
- `analyze.py` - Image analysis with local contrast detection and wire constraint validation
- `triangulate.py` - 3D position calculation from multiple camera angles
- `coordinates/` - Generated 3D LED coordinates (CSV, JSON, compact formats)

## HTTP API

Port 8080. Mutating endpoints need `Authorization: Bearer <HTTP_TOKEN>` (or `X-Tree-Token`); read-only ones are open. Source of truth: `firmware/app/api.py` and `firmware/app/calib.py`.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/`, `/status` | GET | | Status: version, slot, boot/trial state, effect, Wi-Fi, MQTT, update, supervisor |
| `/health` | GET | | `{"ok", "healthy", "problems"}` |
| `/log`, `/mqtt`, `/frame`, `/effects`, `/gc` | GET | | Recent log, MQTT client state, last wire frame (base64), effect list |
| `/state` | POST | yes | JSON `{power, brightness, effect, speed, auto_s, params}` |
| `/effects/<name>` | PUT/DELETE | yes | Hot-push (validated, no reboot) or remove an effect module |
| `/files/coords.txt` | PUT | yes | Coordinates in the contract frame (`coords_compact.txt` for legacy files) |
| `/update` | POST | yes | Firmware bundle; `?dry_run=1` verifies only. Reboots into it as a trial |
| `/update/status` | GET | | Progress of the current or last update |
| `/update/github?channel=main\|stable` | POST | yes | Self-update from GitHub Releases (needs `UPDATE_KEY`) |
| `/update/github/check` | GET | yes | Latest version on a channel |
| `/update/confirm`, `/update/rollback` | POST | yes | Confirm a trial now / boot the previous slot |
| `/reboot` | POST | yes | Reboot |
| `/calibrate`, `/calibrate/status`, `/calibrate/log` | GET | | Calibration status and playback timing log |
| `/calibrate/start`, `/calibrate/stop`, `/calibrate/led?n=X`, `/calibrate/binary?bit=X`, `/calibrate/clear` | GET/POST | yes | Lock (TTL), unlock, legacy static frames |
| `/calibrate/sequence`, `/calibrate/play`, `/calibrate/halt`, `/calibrate/frame` | POST | yes | `.tcal` upload, timed playback, static frame |

mDNS: the hostname is set before Wi-Fi starts so the rp2 lwIP responder answers for `tree.local`.

### Updates

CI publishes GitHub Release assets (`tree-firmware.bin`, `manifest.json`) only for trusted refs of this repository: every push to `main` updates the rolling prerelease `firmware-main` (channel `main`), and tags `v*` create releases (channel `stable`). Bundles are signed with the `FIRMWARE_SIGNING_KEY` Actions secret when it is set; a device with the same string as `UPDATE_KEY` refuses unsigned bundles and can update itself from GitHub (TLS on the device is not certificate-checked, so the signature is what makes that safe). Without a key, use `./flash.py --ota --github`, which downloads over verified HTTPS on the host.

## MQTT / Home Assistant

Discovery puts every entity under one device (identifier `xmas_tree_pico`, so the old light entity carries over):

- light (JSON schema, brightness, effects incl. `auto`); numbers for speed, auto-cycle duration and every effect param; update entity; buttons for reboot and calibration; diagnostic sensors (RSSI, free heap, uptime, FPS, power cap, reset cause, boot status, IP, last effect error) and binary sensors (power cap engaged, calibrating).
- Topics under `xmas_tree/`: `availability` (online/offline, last will), `state` + `set` (light), `settings`, `diag`, `update`, `cmd/...`, `calibrate/set` + `calibrate/state`.
- Persisted state wins at boot: retained messages on command topics are ignored. HA's birth message (`homeassistant/status` = online) triggers re-discovery.

**Power limiting:** the per-frame cap (default 13% of all LEDs at full white) is applied to the raw frame before brightness, so brightness 100% means "as bright as the PSU budget allows" and the slider always dims.

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
