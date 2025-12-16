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

The tree supports volumetric 3D animations using camera-calibrated LED positions:

```bash
# 1. Capture LED positions from multiple angles
./capture.py --angle front --host 192.168.2.149

# 2. Analyze images to extract 2D positions
./analyze.py --angle front --validate

# 3. Triangulate 3D coordinates from two orthogonal views
./triangulate.py --angles front,side

# 4. Upload coordinates to Pico
curl -X POST --data-binary @coordinates/coords_compact.txt \
  "http://192.168.2.149:8080/update?file=coords_compact.txt"
```

## Roadmap

- [x] 3D coordinate mapping for volumetric animations
- [ ] RGB color picker in Home Assistant
