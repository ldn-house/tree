# Christmas Tree LED Controller

RGB LED strip controller for a 6ft Christmas tree with 500 WS2811 LEDs, inspired by [Matt Parker's 3D Christmas tree](https://www.youtube.com/watch?v=TvlpIojusBE).

## Hardware

- Pi Pico 2W running MicroPython
- 500x WS2811 12V RGB LEDs (GRB color order)
- Logic level shifter (3.3V → 5V for data signal)
- 12V power supply for LEDs

## Usage

```bash
# Flash to Pico (saves as main.py, auto-runs on boot)
./flash.py

# First-time setup (all scripts: main, ota, mqtt, config)
./flash.py --all

# Push updates via WiFi OTA (no USB needed)
./flash.py --ota --host 192.168.2.149

# Preview animations in browser
./serve.py
```

uv automatically handles dependencies.

## Home Assistant Integration

The tree integrates with Home Assistant via MQTT, appearing as a light entity with:
- Power on/off
- Brightness control (with automatic power limiting)
- Effect selection (auto-cycle or lock to specific effect)

### Setup

1. Install Mosquitto broker add-on in Home Assistant
2. Add MQTT credentials to `config.py` (see `config.example.py`)
3. Flash with `./flash.py --all`
4. Tree auto-discovers in Home Assistant

### Power Limiting (HDR)

The tree automatically limits total power draw per-frame to prevent PSU brownouts. Sparse animations (sparkle) can be brighter than dense ones (rainbow). Brightness slider goes 0-100% but actual output is capped based on how many LEDs are lit.

## Animations

- **auto** - Cycles through all effects
- Rainbow cycle
- Christmas (red/green theater chase)
- White sparkle
- Candy cane stripes
- Multicolor twinkle
- Warm white (solid)
- **Plane sweep** - Horizontal plane moves up/down tree (uses 3D coordinates)

## 3D Coordinate Mapping

The tree supports volumetric 3D animations using camera-calibrated LED positions. `calibrate.py` is the current toolkit; see [docs/calibration-findings.md](docs/calibration-findings.md) for why, and for benchmark numbers.

**Video (recommended, needs the calibration protocol in firmware).** The tree loops a short coded sequence ([docs/calibration-protocol.md](docs/calibration-protocol.md)). Film it for about 11 s from each of 4–6 spots around the tree, front view first:

```bash
./calibrate.py decode-video front.mp4 right.mp4 back.mp4 left.mp4 --out positions
./calibrate.py solve positions/{front,right,back,left}_detections.json --out coordinates
```

**Still captures (works with the current firmware).**

```bash
# 1. Capture each LED from 4 or more angles (front first)
./capture.py --angle front --host 192.168.2.149

# 2. Detect LEDs (captures made before capture_version 2 show LED i-1 in led_i: use --lag 1)
./calibrate.py detect --views front,right,back,left

# 3. Triangulate with perspective bundle adjustment -> coordinates/coords.txt
./calibrate.py solve positions/{front,right,back,left}_detections.json --out coordinates
```

`coords.txt` has one `x,y,z` (or `null`) line per LED: y from 0 (lowest LED) to 1 (highest), x and z in the same units, origin on the trunk, +x towards the front view. `coords_metres.csv` adds metres, per-LED confidence and the number of views. After a solve, `./calibrate.py export-legacy` regenerates the legacy `coords_compact.txt` and `coords.json` that older tools read; [coordinates/README.md](coordinates/README.md) describes every file. The older `analyze.py` and `triangulate.py` still work, but writing their output over the calibrated files would replace them.

The synthetic benchmark (`./calibrate.py simulate`, `simulate-video`, `bench-stills`, `bench-video`) renders captures with known ground truth so any pipeline can be scored.

## Roadmap

- [x] 3D coordinate mapping for volumetric animations
- [ ] RGB color picker in Home Assistant
