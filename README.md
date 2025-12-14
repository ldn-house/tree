# Christmas Tree LED Controller

RGB LED strip controller for a 6ft Christmas tree, inspired by [Matt Parker's 3D Christmas tree](https://www.youtube.com/watch?v=TvlpIojusBE).

## Current Setup

- Raspberry Pi with `rpi_ws281x` library
- WS2811/WS2812 LED strip (planning for 500 LEDs)
- SPI MOSI (GPIO 10) for data

## Usage

**Raspberry Pi (current):**
```bash
sudo python script.py
```

**Pi Pico 2W (with uv):**
```bash
# Flash to Pico (saves as main.py, auto-runs on boot)
./flash.py

# Or just run without flashing
./flash.py --run
```

uv automatically handles the mpremote dependency.

## Roadmap

- [ ] Migrate to MicroPython on Pi Pico 2W (PIO for precise timing)
- [ ] Add more animation patterns
- [ ] 3D coordinate mapping for volumetric animations
