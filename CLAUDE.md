# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Christmas tree LED controller for 500 WS2811 12V LEDs on a 6ft tree. Inspired by Matt Parker's 3D Christmas tree project. Goal is volumetric 3D animations once coordinate mapping is implemented.

## Hardware

- **Target**: Pi Pico 2W running MicroPython 1.27.0 (RPI_PICO2_W; bundles are mpy v6.3 bytecode from mpy-cross 1.27). Measured on the board: 452,544 bytes heap free at an idle REPL, 2,621,440-byte filesystem
- **LEDs**: 500x WS2811 12 V 12 mm bullet pixels on 3-wire strands (JST-SM joins, power injected at the joins), RGB on the wire (`COLOR_ORDER`). About 0.3 W per pixel at full white (to be measured): ~150 W, ~12.5 A for all 500. PSU: 12 V 20 A.
- **Data signal**: Pico GP0 (3.3 V) → 4-channel BSS138 level shifter (10 kΩ pull-ups; LV = 3V3, HV = Pico VBUS 5 V) → first pixel, driven by a PIO state machine fed by DMA. The shifter's high side rises only through its pull-up (RC ≈ 0.5 µs with the cable), so short high pulses can vanish: the default `LED_TIMING_NS` is MicroPython neopixel's (400, 850, 800, 450) ns, which drove this hardware before.
- **Pico power**: from a USB hub on its own 5 V adapter; ground is shared with the strip through the level shifter.

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
uv run --with pytest --with paho-mqtt --with mpy-cross==1.27.0.post2 pytest tests/firmware
```

Run the tests on the board's MicroPython version (1.27.0, as CI does): set `MICROPYTHON` to a 1.27 unix build, because newer ports have APIs the board lacks (1.27's sockets have no `sendall()`). The board-sized heap test (`test_heap.py`) also needs a 32-bit build in `MICROPYTHON32` (`make -C ports/unix MICROPY_FORCE_32BIT=1`; on macOS, run `tests/firmware/heapcheck.py` in a `linux/386` container).

OTA needs `HTTP_TOKEN` (config.py, `$TREE_TOKEN` or `--token`). `./flash.py --ota --config` replaces config.py over Wi-Fi (sent as JSON values, signed with `UPDATE_KEY`); flash.py validates config.py before sending it anywhere. The first USB install also removes the old single-slot firmware files (`ota.py`, `mqtt.py`, `selfupdate.py`, `github.py`, `version.py`, `lib/umqtt`, `lib/ssl.mpy`) and keeps `config.py` and the root `coords_compact.txt`.

## Architecture

Firmware lives in `firmware/`:

- **Recovery layer** (installed over USB; no update, upload or API path writes it): `boot.py`, `main.py`, `bootmgr.py`, `safemode.py`. See "Trust model" for what that does and doesn't protect against.
  - `bootmgr.py` keeps `/boot.json` (active slot, previous slot, trial flag, boot counter, generation floor). Bundles stream into staging (`/fw/new`), are verified (size, SHA-256, signature, generation, `.mpy` version, Python syntax), and only then replace the inactive slot `/fw/a` or `/fw/b`; switching is one atomic rename of `boot.json`. A rejected upload or a dry-run never touches either slot, and a real install is refused while the running firmware is an unconfirmed trial.
  - A freshly installed slot is a *trial*: if the device reboots before the app reports healthy, the next boot rolls back. A confirmed slot that fails 3 boots in a row falls back to the previous slot, then to **safe mode** (`safemode.py`: Wi-Fi + minimal authenticated HTTP server that accepts a new bundle or a new config). If the boot can't even be recorded (flash full or failing), main.py goes straight to safe mode, which first reclaims staging, the 256 KB recovery reserve (`/fw/reserve.bin`), hot-pushed effects and logs.
  - Anti-replay: each bundle's manifest carries a signed `generation` (build time). Once a generation is confirmed healthy it becomes the floor, and older bundles are refused unless the request says `allow_downgrade=1` (`flash.py --allow-downgrade`; USB installs always may). Self-update from GitHub never downgrades.
  - Config: the last config.py that loaded is kept as `/config.good.json`; if config.py breaks, both the app and safe mode use it, so the tree stays reachable and `PUT /config` can fix it.
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

Flash layout on the device: recovery files and `config.py` at `/`, slots in `/fw/a` and `/fw/b`, staging in `/fw/new`, the recovery reserve `/fw/reserve.bin`, hot-pushed effects in `/fx`, plus `/boot.json`, `/config.good.json`, `/state.json`, `/coords.txt`.

### Trust model

- Updates can't brick the tree: truncated, corrupt, wrongly signed, replayed or otherwise rejected bundles change nothing, and a bundle that installs but doesn't come up healthy is rolled back automatically.
- Code you choose to run is trusted. Firmware bundles and hot-pushed effects execute with full access to the flash (MicroPython has no sandbox), so they *could* overwrite the recovery layer or erase the filesystem; that would need USB to fix. With `UPDATE_KEY` set, bundles, effects and config uploads must all be signed with it, so the HTTP token alone can't run code. Without `UPDATE_KEY`, the HTTP token is full administrative access, including code execution.
- A frozen recovery layer (custom MicroPython UF2) would make the recovery code itself unmodifiable from Python; it is not implemented (see PR #21's discussion of review item H4).

### Calibration Tools (for 3D coordinate mapping)

- `capture.py` - Camera capture script for LED position detection
- `analyze.py` - Image analysis with local contrast detection and wire constraint validation
- `triangulate.py` - 3D position calculation from multiple camera angles
- `coordinates/` - Generated 3D LED coordinates (CSV, JSON, compact formats)

## HTTP API

Port 8080. Mutating endpoints need `Authorization: Bearer <HTTP_TOKEN>` (or `X-Tree-Token`); read-only ones are open. Source of truth: `firmware/app/api.py` and `firmware/app/calib.py`.

Signatures (`X-Signature`) are hex HMAC-SHA256 with `UPDATE_KEY` over `b"tree-<kind>\n" + message`: kind `effect` with message `name + "\n" + module bytes`, kind `config` with the JSON body. Request heads are bounded (1 KB per line, 4 KB in total) and every request has a whole-request deadline.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/`, `/status` | GET | | Status: version, slot, boot/trial state, effect, Wi-Fi, MQTT, update, supervisor |
| `/health` | GET | | `{"ok", "healthy", "problems"}` |
| `/log`, `/mqtt`, `/frame`, `/effects`, `/gc` | GET | | Recent log, MQTT client state, last wire frame (base64), effect list |
| `/state` | POST | yes | JSON `{power, brightness, effect, speed, auto_s, params}` |
| `/effects/<name>` | PUT/DELETE | yes | Hot-push (validated, no reboot; `X-Signature` required when `UPDATE_KEY` is set) or remove an effect module |
| `/config` | PUT | yes | Replace config.py from JSON values (no code accepted); `X-Signature` required when `UPDATE_KEY` is set; reboots |
| `/files/coords.txt` | PUT | yes | Coordinates in the contract frame (`coords_compact.txt` for legacy files) |
| `/update` | POST | yes | Firmware bundle; `?dry_run=1` verifies only, `?allow_downgrade=1` accepts an older generation. Reboots into it as a trial |
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

**Power limiting:** each frame is scaled so the estimated strip current at 12 V is at most `POWER_LIMIT_A` (default 1.5 A = 18 W, about 12% of full white), using `LED_MA_PER_CHANNEL` (default 8 mA per colour channel at full duty, ~0.3 W per pixel at white; to be measured). The strip is fed once, at its data-input end, so the binding limits are the first JST-SM connector (~3 A) and voltage drop along ~50 m of thin strand wire, not the 20 A PSU: raise `POWER_LIMIT_A` only after fused power-injection runs are installed and the far end has been checked for dimming. The cap applies to the raw frame before brightness, so brightness 100% means "as bright as the limit allows" and the slider always dims. `GET /` → `power` and the HA sensors "LED current (estimated, 12 V)" (A) and "Power cap engaged" report it. `POWER_BUDGET_PERCENT` is a deprecated alias.

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
