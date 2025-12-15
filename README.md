# Christmas Tree LED Controller

RGB LED strip controller for a 6ft Christmas tree with 500 WS2811 LEDs, inspired by [Matt Parker's 3D Christmas tree](https://www.youtube.com/watch?v=TvlpIojusBE).

## Hardware

- Pi Pico 2W running MicroPython
- 500x WS2811 12V RGB LEDs
- Logic level shifter (3.3V → 5V for data signal)
- 12V power supply for LEDs

## Usage

```bash
# Flash to Pico (saves as main.py, auto-runs on boot)
./flash.py

# Or just run without flashing
./flash.py --run
```

uv automatically handles the mpremote dependency.

## Animations

- Rainbow cycle
- Theater chase (red/green)
- White sparkle
- Comet with tail (blue/gold)
- Candy cane stripes
- Multicolor twinkle

## Roadmap

- [ ] 3D coordinate mapping for volumetric animations
