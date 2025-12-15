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

# First-time setup (includes WiFi config and MQTT)
./flash.py --config

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
3. Flash with `./flash.py --config`
4. Tree auto-discovers in Home Assistant

### Power Limiting (HDR)

The tree automatically limits total power draw per-frame to prevent PSU brownouts. Sparse animations (comet, sparkle) can be brighter than dense ones (rainbow). Brightness slider goes 0-100% but actual output is capped based on how many LEDs are lit.

## Animations

- **auto** - Cycles through all effects
- Rainbow cycle
- Theater chase (red/green/christmas)
- White sparkle
- Comet with tail (blue/gold)
- Candy cane stripes
- Multicolor twinkle
- Warm white (solid)

## Roadmap

- [ ] 3D coordinate mapping for volumetric animations
- [ ] RGB color picker in Home Assistant
