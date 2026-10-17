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

# Update over Wi-Fi from a local build (waits for the device to confirm it)
./flash.py --ota --host tree.local

# ...or from the latest GitHub Release
./flash.py --ota --host tree.local --github

# Preview animations in browser
./serve.py
```

uv automatically handles dependencies.

### Updates that can't brick the tree

The firmware has two slots. An update is written to the unused slot, verified (sizes, SHA-256 hashes, optional signature, compile checks) and booted as a trial; if it doesn't come up healthy it is rolled back automatically. If nothing will boot, the tree drops into a safe mode that still accepts a new bundle over Wi-Fi. The small recovery layer that does this is only ever written over USB. See CLAUDE.md for details.

## Home Assistant Integration

The tree integrates with Home Assistant via MQTT discovery, as one device with:
- A light: power, brightness (with automatic power limiting) and effect (auto-cycle or lock to one effect)
- Settings: speed, auto-cycle duration and each effect's parameters
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

The tree limits total power draw per frame to prevent PSU brownouts. The limit is applied before brightness, so 100% brightness means "as bright as the power budget allows" and lower settings always dim. Sparse animations (sparkle) can be brighter than dense ones (rainbow).

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
TREE_TOKEN=... ./capture.py --angle front --host 192.168.2.149

# 2. Analyze images to extract 2D positions
./analyze.py --angle front --validate

# 3. Triangulate 3D coordinates from two orthogonal views
./triangulate.py --angles front,side

# 4. Upload coordinates to Pico
./flash.py --ota --host tree.local --coords
```

## Roadmap

- [x] 3D coordinate mapping for volumetric animations
- [ ] RGB color picker in Home Assistant
