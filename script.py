from rpi_ws281x import PixelStrip, Color, ws
import time

LED_COUNT = 150
LED_PIN = 10  # GPIO 10 = SPI MOSI
LED_FREQ_HZ = 800000
LED_DMA = 10
LED_BRIGHTNESS = 64  # 25% brightness
LED_INVERT = False
LED_CHANNEL = 0
LED_STRIP = ws.WS2811_STRIP_RGB  # WS2811 uses RGB order, not GRB

strip = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL, LED_STRIP)
strip.begin()

def wheel(pos):
    """Generate rainbow colors across 0-255 positions."""
    if pos < 85:
        return Color(pos * 3, 255 - pos * 3, 0)
    elif pos < 170:
        pos -= 85
        return Color(255 - pos * 3, 0, pos * 3)
    else:
        pos -= 170
        return Color(0, pos * 3, 255 - pos * 3)

def rainbow_cycle(wait_ms=20):
    """Rainbow that moves along the strip."""
    for j in range(256):
        for i in range(LED_COUNT):
            strip.setPixelColor(i, wheel((i * 256 // LED_COUNT + j) & 255))
        strip.show()
        time.sleep(wait_ms / 1000.0)

print("Running rainbow cycle... Ctrl+C to stop")
try:
    while True:
        rainbow_cycle(20)
except KeyboardInterrupt:
    for i in range(LED_COUNT):
        strip.setPixelColor(i, Color(0, 0, 0))
    strip.show()
    print("\nLEDs off")
