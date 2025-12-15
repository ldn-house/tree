"""
Christmas Tree LED Controller - Pi Pico 2W
MicroPython with PIO-based NeoPixel driver
"""

from machine import Pin
from neopixel import NeoPixel
import time
import random

LED_COUNT = 500
LED_PIN = 0
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


def clear():
    """Turn off all LEDs."""
    for i in range(LED_COUNT):
        strip[i] = (0, 0, 0)
    strip.write()


def fill(color):
    """Fill all LEDs with a single color."""
    for i in range(LED_COUNT):
        strip[i] = scale(color)
    strip.write()


# ─── Animations ───────────────────────────────────────────────────────────────


def rainbow_cycle(wait_ms=20, cycles=1):
    """Rainbow that moves along the strip."""
    for _ in range(cycles):
        for j in range(256):
            for i in range(LED_COUNT):
                strip[i] = scale(wheel((i * 256 // LED_COUNT + j) & 255))
            strip.write()
            time.sleep_ms(wait_ms)


def chase(color, wait_ms=50, cycles=3):
    """Theater chase animation."""
    for _ in range(cycles):
        for offset in range(3):
            for i in range(LED_COUNT):
                strip[i] = scale(color) if i % 3 == offset else (0, 0, 0)
            strip.write()
            time.sleep_ms(wait_ms)


def sparkle(color, wait_ms=50, density=0.05, duration_ms=5000):
    """Random twinkling lights."""
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        clear()
        for i in range(LED_COUNT):
            if random.random() < density:
                strip[i] = scale(color)
        strip.write()
        time.sleep_ms(wait_ms)


def comet(color, tail_length=20, wait_ms=20, cycles=2):
    """Comet with fading tail."""
    for _ in range(cycles):
        for head in range(LED_COUNT + tail_length):
            for i in range(LED_COUNT):
                distance = head - i
                if 0 <= distance < tail_length:
                    fade = 1.0 - (distance / tail_length)
                    strip[i] = scale(tuple(int(c * fade) for c in color))
                else:
                    strip[i] = (0, 0, 0)
            strip.write()
            time.sleep_ms(wait_ms)


def candy_cane(wait_ms=100, cycles=50):
    """Red and white stripes that move."""
    stripe_width = 10
    for offset in range(cycles):
        for i in range(LED_COUNT):
            if ((i + offset) // stripe_width) % 2 == 0:
                strip[i] = scale((255, 0, 0))
            else:
                strip[i] = scale((255, 255, 255))
        strip.write()
        time.sleep_ms(wait_ms)


def twinkle_multi(wait_ms=100, duration_ms=10000):
    """Twinkling multicolor Christmas lights."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 255, 255)]
    # Initialize with random colors
    led_colors = [random.choice(colors) for _ in range(LED_COUNT)]
    led_brightness = [random.random() for _ in range(LED_COUNT)]

    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        for i in range(LED_COUNT):
            # Randomly change some LEDs
            if random.random() < 0.02:
                led_colors[i] = random.choice(colors)
                led_brightness[i] = random.random()
            b = led_brightness[i]
            strip[i] = scale(tuple(int(c * b) for c in led_colors[i]))
        strip.write()
        time.sleep_ms(wait_ms)


def wave(color1, color2, wait_ms=50, cycles=3):
    """Smooth wave between two colors."""
    for _ in range(cycles):
        for offset in range(LED_COUNT):
            for i in range(LED_COUNT):
                ratio = (1 + math.sin((i + offset) * 0.1)) / 2
                r = int(color1[0] * ratio + color2[0] * (1 - ratio))
                g = int(color1[1] * ratio + color2[1] * (1 - ratio))
                b = int(color1[2] * ratio + color2[2] * (1 - ratio))
                strip[i] = scale((r, g, b))
            strip.write()
            time.sleep_ms(wait_ms)


# ─── Main Loop ────────────────────────────────────────────────────────────────


ANIMATIONS = [
    lambda: rainbow_cycle(20, 2),
    lambda: chase((255, 0, 0), 50, 10),
    lambda: chase((0, 255, 0), 50, 10),
    lambda: sparkle((255, 255, 255), 30, 0.03, 8000),
    lambda: comet((0, 0, 255), 30, 15, 3),
    lambda: comet((255, 215, 0), 30, 15, 3),
    lambda: candy_cane(80, 100),
    lambda: twinkle_multi(80, 10000),
]


print("Christmas tree starting...")
try:
    while True:
        for anim in ANIMATIONS:
            anim()
except KeyboardInterrupt:
    clear()
    print("LEDs off")
