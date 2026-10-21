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

# Build the bundle only (dist/tree-firmware.bin, dist/manifest.json; firmware/app + effects/)
tools/build_firmware.py

# Hot-push effect modules to a running tree (no firmware update, no reboot)
./push_effect.py <name> --host tree.local --activate

# Offline firmware tests: MicroPython unix port, mocked hardware, local mosquitto
uv run --with pytest --with paho-mqtt --with mpy-cross==1.27.0.post2 pytest tests/firmware
```

Device tests poll for the outcome they check with a deadline (`devsim.wait_for`, `Bus.wait`, `Device.saved_state()` for debounced saves) rather than sleeping a fixed time and asserting: the test process and the simulated device are scheduled independently, and CI runners are slower and noisier than a laptop. Fixed sleeps are only for absence checks ("no reset for 8 s", "nothing published") and for spacing a scenario. Run the tests on the board's MicroPython version (1.27.0, as CI does): set `MICROPYTHON` to a 1.27 unix build, because newer ports have APIs the board lacks (1.27's sockets have no `sendall()`). The board-sized heap test (`test_heap.py`) also needs a 32-bit build in `MICROPYTHON32` (`make -C ports/unix MICROPY_FORCE_32BIT=1`; on macOS, run `tests/firmware/heapcheck.py` in a `linux/386` container).

In CI, a pull request that changes only `effects/`, `renders/`, `docs/` or top-level Markdown runs just the firmware tests that bundle the repo's effects (`test_real_effects.py`, `test_heap.py`, `test_output_copy.py`) in the `test` job; anything else, and every push to main or a tag, runs the full suite. A newer push to a pull request cancels its running checks; runs on main and tags are never cancelled.

OTA needs `HTTP_TOKEN` (config.py, `$TREE_TOKEN` or `--token`). `./flash.py --ota --config` replaces config.py over Wi-Fi (sent as JSON values, signed with `UPDATE_KEY`); flash.py validates config.py before sending it anywhere. The first USB install also removes the old single-slot firmware files (`ota.py`, `mqtt.py`, `selfupdate.py`, `github.py`, `version.py`, `lib/umqtt`, `lib/ssl.mpy`) and keeps `config.py` and the root `coords_compact.txt`.

## Architecture

Firmware lives in `firmware/`:

- **Recovery layer** (installed over USB; no update, upload or API path writes it): `boot.py`, `main.py`, `bootmgr.py`, `safemode.py`, `cfgcheck.py`. See "Trust model" for what that does and doesn't protect against.
  - `cfgcheck.py` is the config schema (type, range and finiteness of every setting, plus the WS2811 timing validator), shared by the boot manager, safe mode, the app and `flash.py`. A config with any invalid setting is not used.
  - `bootmgr.py` keeps `/boot.json` (active slot, previous slot, trial flag, boot counter, generation floor). Bundles stream into staging (`/fw/new`), are verified (size, SHA-256, signature, generation, `.mpy` version, Python syntax), and only then replace the inactive slot `/fw/a` or `/fw/b`; switching is one atomic rename of `boot.json`. A rejected upload or a dry-run never touches either slot, and a real install is refused while the running firmware is an unconfirmed trial. The one exception is an explicit safe-mode recovery upload with `?reclaim_slot=1`, which may delete the inactive slot to make room, and only if that would actually be enough.
  - A freshly installed slot is a *trial*: if the device reboots before the app reports healthy, the next boot rolls back. A confirmed slot that fails 3 boots in a row falls back to the previous slot, then to **safe mode** (`safemode.py`: Wi-Fi + minimal authenticated HTTP server that accepts a new bundle or a new config). If the boot can't even be recorded (flash full or failing), main.py goes straight to safe mode, which first reclaims staging, the 256 KB recovery reserve (`/fw/reserve.bin`), hot-pushed effects and logs.
  - Anti-replay: each bundle's manifest carries a signed `generation` (build time). Once a generation is confirmed healthy it becomes the floor (kept in `boot.json` and `/fw/floor.json`; if both are lost, the newest installed slot's generation is used, never 0), and older bundles, signed or not, are refused unless the request says `allow_downgrade=1` (`flash.py --allow-downgrade`; USB installs always may). Self-update from GitHub never downgrades.
  - Config: `/config.good.json` is the last config *confirmed to reach the network* (Wi-Fi joined with its own credentials, HTTP serving, MQTT connected if configured); merely importing doesn't promote it. If config.py is missing, broken or fails the schema, the app and safe mode use the good config. `PUT /config` is a trial like firmware: the old config.py is kept as `config.prev.py`, and if the next boot doesn't reach the network with the new config by the trial deadline it is reverted. The trial (with a digest of the new file) is recorded in boot.json before config.py is replaced, so a power cut at any point leaves either the old config or the new one on a trial; a boot that finds config.py isn't the recorded file drops the change. The replaced config's Wi-Fi credentials are a fallback (after the good snapshot) for the app and safe mode until a change is confirmed. Meanwhile Wi-Fi alternates to the last good credentials, so a wrong SSID or password never strands the tree. `UPDATE_KEY` can't be changed or removed over the network (USB only).
  - Reserve: `/fw/reserve.bin` (256 KB) is (re)built in the background 30 s after boot and every 10 minutes, resuming a partial file; `GET /` → `reserve` reports whether it is complete. A build isn't started during calibration playback, and one already running pauses until playback ends, as do debounced state saves: a flash erase stops all code on the Pico for tens of ms, and playback frames are timed to the millisecond.
  - `boot.py` only keeps an already-running watchdog fed by a timer, so mpremote sessions (which soft-reset without running main.py) are never cut off. The boot counter is in main.py's path.
  - Recovery API: `bootmgr.API` (3) is what the installed recovery layer provides; a bundle's manifest `bootmgr` is the oldest API its app runs on (`RECOVERY_API` in `firmware/app/__init__.py`, currently 2), and an older recovery layer refuses the bundle before staging. Since OTA never updates the recovery layer, the app uses newer bootmgr functions only through `app/recovery.py`, which falls back to its own implementation on an older layer and reports `recovery layer outdated (USB install recommended)` in `GET /` → `recovery`, `/health` → `warnings` and HA's "Recovery layer" sensor, without failing health. Raise `RECOVERY_API` only when the app can't fall back. The other direction holds too: launch info keeps the API 2 contract (`reset_cause` is the machine-level cause, `wdt` after any watchdog reset, which API 2 apps quarantine on) and adds `reset_kind` (`watchdog` / `reboot`) for newer apps. `tests/firmware/test_recovery_compat.py` OTAs HEAD onto 52cda75's recovery layer, and runs 52cda75's app on this one (after a rollback) through a hang.
- **App** (`firmware/app/`, one slot, replaced by updates): a single asyncio event loop, no threads.
  - `runtime.py` wires tasks together; `supervisor.py` owns the 8 s hardware watchdog and feeds it only while every task heartbeats, decides health, confirms trial updates and performs reboots. It collects garbage every 10 s and reports the free heap right after that (`supervisor.heap_live`, `heap_live_min`; HA "Free memory"). The runtime also sets `gc.threshold(32 KB)`, so a collection runs after every 32 KB allocated instead of only when the heap is exhausted; the raw low-water mark `supervisor.heap_low_uncollected` therefore stays within about 32 KB of the live figure, and `heap_sites` holds the lowest free heap after each effect start, MQTT publish and HTTP request (a new low under 64 KB is logged). `tests/firmware/heapcheck.py` requires ≥ 64 KB uncollected through three auto-cycle loops with MQTT connected, on a 400 KB 32-bit heap.
  - `trace.py`: each task marks its phase (effect render, output stage, effect start, mqtt, http request, state save, gc, …) in watchdog scratch 1, which survives a watchdog reset, and a 1 s soft timer adds how long the event loop has been stalled (Python still running) or not (blocked in native code). After a watchdog expiry the supervisor didn't decide on, `GET /` → `boot.hang`, `boot.reset_reason` ("hung in …"), the log and HA's "Last reset cause" say where the loop was. Turns of the event loop longer than 1.5 s are logged (`supervisor.stall_max_ms`, `stall_in`).
  - `engine.py` runs effects from the `effects/` package (docs/effect-api.md): render → output stage (`effects/_output.py`: power cap on the raw frame, brightness, gamma, colour order) → `driver.py` (PIO + DMA). Effect exceptions mark the effect failed and the scheduler moves on; an effect that hangs is quarantined for the next boot (watchdog scratch register + `/fx_probe.txt`), but only after a genuine watchdog expiry. On the rp2, `machine.reset()` is `watchdog_reboot()`, so `machine.reset_cause()` says `WDT_RESET` after every deliberate reset too; `bootmgr.reset_cause()` reads the watchdog REASON register (TIMER = expired → `watchdog`, FORCE = forced → `reboot`, neither → `pwron`), and every deliberate reset path, including Ctrl-C at the USB REPL, clears the hang markers first (`bootmgr.deliberate()`; on an API 2 recovery layer, `app/recovery.py` does both, best effort). Status and HA's "Last reset cause" show `watchdog`, `reboot` or `pwron`.
  - `state.py`: power, brightness, effect, speed, auto-cycle duration and params, persisted to `/state.json` and authoritative at boot.
  - `net.py` (Wi-Fi state machine), `dns.py` (async DNS), `mqtt.py` (async MQTT 3.1.1 client with LWT, PINGREQ every keepalive/2 whatever else is sent, dead-peer detection after 1.5 × keepalive of silence, backoff), `ha.py` (Home Assistant discovery and commands).
  - `httpd.py` + `api.py`: HTTP API with token auth; `updates.py`: bundle uploads and GitHub Releases self-update; `calib.py`: calibration lock and sequence player (docs/calibration-protocol.md), with the vendored `seqfile.py` codec. Finite playback is never cut by the lock's TTL; an infinite one (`loops=0`, as HA's "Play calibration sequence" button sends) runs until its `ttl_s`.
- `tools/build_firmware.py` builds the bundle (mpy-cross compiles `firmware/app` and `effects/`, adds `data/effects.json` metadata, legacy coordinates and the default calibration sequence, signs with `FIRMWARE_SIGNING_KEY`).
- `flash.py` — uv script (PEP 723) to install over USB (mpremote) or OTA.
- `tests/firmware/` — the firmware on the MicroPython unix port: `mocks/` (machine, rp2, network), `devsim.py` (simulated resets and watchdog), unit tests in `mp/`, and device tests for updates, faults, MQTT and calibration. Device tests use the deliberately small `fixtures/effects/` package (fast and predictable); `test_real_effects.py` and the heap campaign build the repo's `effects/` the way `flash.py` does.
- `config.py` — WiFi, token, MQTT, signing key (gitignored; see `config.example.py`)

Effects, previews and tooling:

- `effects/` - Pure effect modules plus the shared output stage (`_output.py`) and coordinate loading (`_coords.py`); see `docs/effect-api.md`. `EFFECTS` is the auto-cycle order; modules in `DISABLED` are kept in the tree but not bundled.
- `viewer/` - Browser viewer: three.js tree running `effects/` in Pyodide; `serve.py` serves it locally (with hot reload)
- `render.py` - Headless effect renderer (GIF, contact sheet, metrics)
- `smoke.py` - Validates every effect (CI: `.github/workflows/effects.yml`)
- `new_effect.py` / `push_effect.py` - Scaffold an effect; hot-push effect files to the tree (`PUT /effects/<name>`)
- `tools/` - `build_firmware.py` (above), `mp_check.py` (MicroPython allocation check), `mock_tree.py` (local stand-in for the firmware's effects API)

Flash layout on the device: recovery files and `config.py` at `/`, slots in `/fw/a` and `/fw/b`, staging in `/fw/new`, the recovery reserve `/fw/reserve.bin`, hot-pushed effects in `/fx`, plus `/boot.json`, `/config.good.json`, `/state.json`, `/coords.txt`.

### Trust model

- Updates can't brick the tree: truncated, corrupt, wrongly signed, replayed or otherwise rejected bundles change nothing, and a bundle that installs but doesn't come up healthy is rolled back automatically.
- Code you choose to run is trusted. Firmware bundles and hot-pushed effects execute with full access to the flash (MicroPython has no sandbox), so they *could* overwrite the recovery layer or erase the filesystem; that would need USB to fix. With `UPDATE_KEY` set, bundles, effects and config uploads must all be signed with it, so the HTTP token alone can't run code. Without `UPDATE_KEY`, the HTTP token is full administrative access, including code execution.
- Replay: firmware bundles have anti-replay (the generation floor). Signed effects and signed configs do not: a signature stays valid indefinitely, so an old signed effect or config can be re-sent by someone holding the HTTP token. Configs still go through the schema and the network trial, and can't change `UPDATE_KEY`; effects run as code the owner once signed. Rotate `UPDATE_KEY` over USB to revoke everything signed with the old one.
- A frozen recovery layer (custom MicroPython UF2) would make the recovery code itself unmodifiable from Python; it is not implemented (see PR #21's discussion of review item H4).

### Calibration Tools (for 3D coordinate mapping)

- `calibrate.py` - uv script with subcommands: `sequence`, `simulate`, `simulate-video`, `detect`, `decode-video`, `solve`, `export-legacy`, `quality`, `bench-stills`, `bench-video`, `explainer`
- `calib/` - library behind it: `.tcal` sequence format (`seqfile.py`, MicroPython-compatible; vendored unchanged as `firmware/app/seqfile.py`), codes and sequences, camera simulator (`render.py`, `synth.py`, `video.py`), detectors (`detect.py`), bundle adjustment (`ba.py`, `solve.py`), scoring (`metrics.py`, `bench.py`), coordinate output (`output.py`)
- `capture.py` - Webcam capture of one LED per frame via `/calibrate/*` (capture_version 2 fixes the stale-frame lag)
- `analyze.py`, `triangulate.py` - Older pipeline (orthographic 0/90/180/270 views), superseded by `calibrate.py`
- `sequences/default.tcal` - Calibration sequence the firmware plays (bundled as `data/calib_default.tcal`); `docs/calibration-protocol.md` is the firmware contract
- `docs/calibrate-the-tree.md` - How to calibrate (filming guide, then the commands); `docs/calibration-findings.md` - method, benchmark results and the first real calibration
- `coordinates/` - Calibrated coordinates: `coords.txt` (contract frame, loaded first everywhere), `coords_metres.csv`, `calibration_report.json`; `coords_compact.txt` and `coords.json` are generated from `coords.txt` by `export-legacy` (see `coordinates/README.md`)

Calibration tests: `uv run --with pytest --with numpy --with scipy --with opencv-python pytest tests/test_seqfile.py tests/test_solve.py`.

## HTTP API

Port 8080. Mutating endpoints need `Authorization: Bearer <HTTP_TOKEN>` (or `X-Tree-Token`); read-only ones are open. Source of truth: `firmware/app/api.py` and `firmware/app/calib.py`.

Signatures (`X-Signature`) are hex HMAC-SHA256 with `UPDATE_KEY` over `b"tree-<kind>\n" + message`: kind `effect` with message `name + "\n" + module bytes`, kind `config` with the JSON body. Request heads are bounded (1 KB per line, 4 KB in total) and every request has a whole-request deadline.

| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/`, `/status` | GET | | Status: version, slot, boot/trial state, effect, Wi-Fi, MQTT, update, supervisor |
| `/health` | GET | | `{"ok", "healthy", "problems", "warnings"}` (warnings, e.g. an outdated recovery layer, don't fail health) |
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

- light (JSON schema, brightness, effects incl. `auto`); numbers for speed, auto-cycle duration and every effect param; update entity; buttons for reboot and calibration; diagnostic sensors (RSSI, free heap, uptime, FPS, power cap, reset cause, boot status, recovery layer, IP, last effect error) and binary sensors (power cap engaged, calibrating).
- Topics under `xmas_tree/`: `availability` (online/offline, last will), `state` + `set` (light), `settings`, `diag` (every 30 s), `effect_running` (retained, on every change: the effect name, `off` or `calibration`; at most once a second) with `effect_running/attributes` (`mode`: auto/locked/fallback/off/calibration, `selected`, `next_switch_s`), `update`, `cmd/...`, `calibrate/set` + `calibrate/state`.
- Persisted state wins at boot: retained messages on command topics are ignored. HA's birth message (`homeassistant/status` = online) triggers re-discovery.
- Availability: the last will is `offline` (retained), but the firmware sends DISCONNECT whenever it drops a session itself, so the will only fires when the tree is really gone. It also subscribes to its own availability topic: `online` is re-published 1.5 s after each connect and whenever `offline` appears while connected (an older session's will arriving late).
- Discovery configs are retained and only re-published on the first connect, when they change (sha256 over all of them; a pass interrupted by a dropped session isn't recorded, so the next connect sends it all again), on HA's birth message, after an effect push, or when the broker has lost its retained messages (our retained availability isn't replayed within 1 s of subscribing). Every reconnect re-publishes all states (retained). `GET /` → `ha` counts `discoveries` and `offline_seen`.
- HA keeps a discovery key that a newer config simply omits (until the MQTT integration is reloaded), so a config that changes how a payload is parsed must state the parsing keys explicitly instead of dropping them: the Running effect sensor sets `value_template: "{{ value }}"` (it once read `{{ value_json.effect }}` from diag), and the light names its `json_attributes_topic` (the old firmware's was `xmas_tree/attributes`). Param numbers render `None` (unknown) when the settings payload lacks their param.
- Each discovery pass first publishes current settings (so new param entities never start from the previous firmware's retained settings) and clears the old single-file firmware's retained `xmas_tree/attributes` with an empty retained message.

**Power limiting:** each frame is scaled so the estimated strip current at 12 V is at most `POWER_LIMIT_A` (default 1.5 A = 18 W, about 12% of full white), using `LED_MA_PER_CHANNEL` (default 8 mA per colour channel at full duty, ~0.3 W per pixel at white; to be measured). The strip is fed once, at its data-input end, so the binding limits are the first JST-SM connector (~3 A) and voltage drop along ~50 m of thin strand wire, not the 20 A PSU: raise `POWER_LIMIT_A` only after fused power-injection runs are installed and the far end has been checked for dimming. The cap applies to the raw frame before brightness, so brightness 100% means "as bright as the limit allows" and the slider always dims. `GET /` → `power` and the HA sensors "LED current (estimated, 12 V)" (A) and "Power cap engaged" report it. `POWER_BUDGET_PERCENT` is a deprecated alias. `LED_MA_PER_CHANNEL` must be positive, and `GAMMA` (applied after the cap) must be ≥ 1, so no accepted setting can push the wire frame above the limit; the reported current is computed from the bytes actually sent.

## Viewer

Preview effects in 3D in the browser without hardware:

```bash
./serve.py  # Opens http://localhost:8000/viewer/
```

`viewer/` loads `effects/` into Pyodide in a Web Worker and calls it through the effect API, running frames through the same output stage (`effects/_output.py`) as the device. It has an effect picker, param sliders, brightness, a current limit in amps, a power-cap indicator showing estimated amps and front/side/top/orbit views. Saving a file in `effects/` reloads it in place. `?effect=<name>` selects an effect from the URL.

## Effects

Effects live in `effects/`, one module per effect, written against the pure API in `docs/effect-api.md`: `Effect(ctx).render(t_ms, frame)` fills a raw RGB bytearray, and the runtime owns timing, brightness, the power cap and the hardware. The same files run on the Pico, under CPython and in Pyodide. Every firmware bundle (`flash.py`, `tools/build_firmware.py`, CI) compiles the registered effects from this directory, plus `data/effects.json` with their metadata; modules in `DISABLED` stay in the repo but aren't shipped. Changing coordinates (`PUT /files/coords.txt`) gives effects a new coordinates list, since `_util` caches per-LED tables per list (`tests/firmware/test_real_effects.py`).

To add an effect, create `effects/<name>.py` with `NAME`, `DURATION_MS`, `PARAMS`, `NEEDS_COORDS` and an `Effect` class, then add the name to `EFFECTS` in `effects/__init__.py`. `render()` must not allocate on MicroPython, so use integer maths over tables precomputed in `__init__` (see `effects/_util.py`).

Scaffold one with `./new_effect.py <name>` (add `--flat` for a template that doesn't need coordinates). Check changes headlessly:

```bash
./render.py <effect>            # renders/<effect>/{sheet.png,anim.gif,metrics.json}
./render.py all                 # every effect + renders/overview.png
./render.py <effect> --watch    # re-render on save
micropython tools/mp_check.py   # every render() is allocation-free on MicroPython
./smoke.py                      # all checks CI runs (.github/workflows/effects.yml)
```

Attach `sheet.png` (and the GIF) to PRs that change an effect.

To try an effect on the tree without a firmware update, use `./push_effect.py <name> --host <ip> --activate`. It reads `HTTP_TOKEN` and `UPDATE_KEY` from `config.py` (like `flash.py`), signs each push, and needs the firmware's effects API; see "Pushing one effect over OTA" in `docs/effect-api.md`.
