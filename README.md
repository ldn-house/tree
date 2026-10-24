# Christmas Tree LED Controller

RGB LED strip controller for a 6ft Christmas tree with 500 WS2811 LEDs, inspired by [Matt Parker's 3D Christmas tree](https://www.youtube.com/watch?v=TvlpIojusBE).

## Hardware

- Pi Pico 2W running MicroPython
- 500x WS2811 12V RGB LEDs (RGB on the wire), driven by PIO + DMA
- Logic level shifter (3.3V → 5V for data signal)
- 12V power supply for LEDs

## Usage

```bash
# First-time setup over USB: recovery layer, firmware, config.py, coordinates
cp config.example.py config.py   # fill in Wi-Fi, HTTP_TOKEN, MQTT
./flash.py --all

# Update over Wi-Fi from a local build (waits for the device to confirm it).
# The bundle holds the firmware and every effect registered in effects/.
./flash.py --ota --host tree.local

# ...or from the latest GitHub Release
./flash.py --ota --host tree.local --github

# Preview effects in the browser (3D viewer with hot reload)
./serve.py

# Render an effect headlessly (GIF, contact sheet, metrics)
./render.py rainbow_cycle

# Export deterministic loops for website playback (no Python in the browser)
./export_web.py --out renders/web --archive renders/tree-effects.tar.gz

# Try an edited effect on the tree without a firmware update (hot-push)
./push_effect.py rainbow_cycle --host tree.local --activate
```

uv automatically handles dependencies.

The [web export contract](docs/web-export.md) documents the manifest, binary playback, decoded preview, and public rolling release download.

### Updates that can't brick the tree

The firmware has two slots. An update is staged, verified (sizes, SHA-256 hashes, signature, an anti-replay generation, compile checks), only then copied over the unused slot, and booted as a trial; if it doesn't come up healthy it is rolled back automatically. A rejected update changes nothing. If nothing will boot, or the flash can't be written, the tree drops into a safe mode that still accepts a new bundle (or a fixed config) over Wi-Fi. The small recovery layer that does this is installed over USB and no update path writes it. Code you install yourself (firmware, hot-pushed effects) is trusted and runs with full access, so set `UPDATE_KEY` to require your signature on all of it. See CLAUDE.md, "Trust model".

## Home Assistant Integration

The tree integrates with Home Assistant via MQTT discovery, as one device with:
- A light: power, brightness (with automatic power limiting) and effect (auto-cycle or lock to one effect)
- Settings: speed, auto-cycle duration, brightness floor, colour accuracy and each effect's parameters
- A firmware update entity, plus reboot and calibration buttons
- Diagnostics: Wi-Fi signal, free memory, uptime, frame rate, power cap, last reset cause, boot status
- Availability: shown unavailable as soon as the tree drops off (MQTT last will)

The tree remembers its power, brightness and effect across reboots and network outages.

### Setup

1. Install Mosquitto broker add-on in Home Assistant
2. Add MQTT credentials to `config.py` (see `config.example.py`)
3. Flash with `./flash.py --all`
4. Tree auto-discovers in Home Assistant

### Power Limiting (HDR)

The tree limits total power draw per frame to prevent PSU brownouts. The limit is applied before brightness, so 100% brightness means "as bright as the power budget allows" and lower settings always dim. The light's 1% is a brightness floor (4% by default, the "Brightness floor" setting): below that, LEDs can't show a colour at all, so the slider maps 1-100% onto floor-100%. At the dim end every LED shows the same colour; the lowest few steps move in whole levels, so neighbouring steps can look the same. A few levels can't show every colour, though: warm white's blue rounds to nothing below about red 10, and it looks yellow. So each effect also gets a colour floor, the lowest level from which its colours stay within the "Colour accuracy" setting (0.01 by default, 0 turns it off), and 1% is whichever floor is higher. Warm white's 1% is about 16%, a rainbow's about 22% and fire's about 27%, while red and green (christmas) stay at 4%. The "Brightness floor in effect" and "Colour floor" sensors show where 1% is. Sparse animations (plane sweep) can be brighter than dense ones (rainbow).

## Animations

- **auto** - Cycles through all effects
- Rainbow cycle
- Christmas (red/green theater chase)
- Candy cane stripes
- Multicolor twinkle
- Warm white (solid)
- **Plane sweep** - Horizontal plane moves up/down tree (uses 3D coordinates)
- **Lighthouse** - Warm beam sweeping round the tree (3D)
- **Raindrops** - Coloured spherical ripples spreading through the tree (3D)
- **Snowfall** - Flakes drifting down through the cone, some settling on branches (3D)

## 3D Coordinate Mapping

The tree supports volumetric 3D animations using camera-calibrated LED positions, made with `calibrate.py`. [docs/calibrate-the-tree.md](docs/calibrate-the-tree.md) is the step-by-step guide (no technical knowledge needed to film), and [docs/calibration-findings.md](docs/calibration-findings.md) has the method and benchmark numbers.

**Video (recommended).** Press **Play calibration sequence** in Home Assistant: the tree loops a short coded blinking pattern ([docs/calibration-protocol.md](docs/calibration-protocol.md)). Film it for about 12 s from each of 4–6 angles, walking round the tree or turning it in front of a fixed phone, front first. Then:

```bash
./calibrate.py decode-video clips/* --out positions/tonight              # clips in parallel
./calibrate.py solve positions/tonight/*_detections.json --out coordinates
./calibrate.py export-legacy coordinates/coords.txt                      # refresh the legacy files
./flash.py --ota --host tree.local --coords                              # install (live, no reboot)
```

**Still captures (one LED per frame, with a webcam).**

```bash
TREE_TOKEN=... ./capture.py --angle front --host tree.local      # repeat for 4+ angles, front first
./calibrate.py detect --views front,right,back,left               # captures before capture_version 2: --lag 1
./calibrate.py solve positions/{front,right,back,left}_detections.json --out coordinates
```

`coordinates/coords.txt` has one `x,y,z` (or `null`) line per LED: y from 0 (lowest LED) to 1 (highest), x and z in the same units, origin on the trunk, +x towards the front view. It's what the firmware, `render.py`, `smoke.py` and the viewer load first; [coordinates/README.md](coordinates/README.md) describes the other files. The older `analyze.py` and `triangulate.py` still run but are superseded; their output would overwrite the calibrated files.

The synthetic benchmark (`./calibrate.py simulate`, `simulate-video`, `bench-stills`, `bench-video`) renders captures with known ground truth, so any pipeline can be scored.

## Roadmap

- [x] 3D coordinate mapping for volumetric animations
- [ ] RGB color picker in Home Assistant
