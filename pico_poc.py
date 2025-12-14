"""
Christmas Tree LED Controller - Pi Pico 2W PoC
MicroPython with PIO-based NeoPixel driver
"""

from machine import Pin
from neopixel import NeoPixel
import time

LED_COUNT = 150  # Change to 500 for full tree
LED_PIN = 0      # GP0 - any GPIO works on Pico
BRIGHTNESS = 0.25

strip = NeoPixel(Pin(LED_PIN), LED_COUNT)


def scale(color):
    """Apply brightness scaling to RGB tuple."""
    return tuple(int(c * BRIGHTNESS) for c in color)


def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return (pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return (255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return (0, pos * 3, 255 - pos * 3)


def rainbow_cycle(wait_ms=20):
    """Rainbow that moves along the strip."""
    for j in range(256):
        for i in range(LED_COUNT):
            strip[i] = scale(wheel((i * 256 // LED_COUNT + j) & 255))
        strip.write()
        time.sleep_ms(wait_ms)


def clear():
    """Turn off all LEDs."""
    for i in range(LED_COUNT):
        strip[i] = (0, 0, 0)
    strip.write()


# Main loop
print("Rainbow cycle running...")
try:
    while True:
        rainbow_cycle(20)
except KeyboardInterrupt:
    clear()
    print("LEDs off")
