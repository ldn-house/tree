"""
Christmas Tree LED Controller - Pi Pico 2W
MicroPython with PIO-based NeoPixel driver + WiFi OTA updates
"""

from machine import Pin
from neopixel import NeoPixel
import time
import random

# OTA imports (optional - works without WiFi)
try:
    import ota
    OTA_AVAILABLE = True
except ImportError:
    OTA_AVAILABLE = False

# MQTT imports (optional - for Home Assistant integration)
try:
    import mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False

LED_COUNT = 500
LED_PIN = 0
BRIGHTNESS = 0.5

# Power limiting: max total RGB sum across all LEDs
# Each LED can draw up to 60mA at full white (255,255,255)
# 500 LEDs * 765 (255*3) = 382,500 max
# Cap at ~15% to be safe with power supply
MAX_POWER = int(LED_COUNT * 255 * 3 * 0.15)

strip = NeoPixel(Pin(LED_PIN), LED_COUNT)

# Current brightness level (updated by MQTT)
_brightness = BRIGHTNESS


def scale(color):
    """Apply brightness scaling and convert RGB to GRB for WS2811."""
    r, g, b = (int(c * _brightness) for c in color)
    return (g, r, b)  # WS2811 uses GRB order


_last_power_log = 0

def safe_write():
    """Write to strip with automatic power limiting (like HDR)."""
    global _last_power_log
    # Calculate total power (sum of all RGB values)
    total = 0
    for i in range(LED_COUNT):
        r, g, b = strip[i]
        total += r + g + b

    if total > MAX_POWER:
        # Scale down proportionally to stay within power budget
        factor = MAX_POWER / total
        for i in range(LED_COUNT):
            r, g, b = strip[i]
            strip[i] = (int(r * factor), int(g * factor), int(b * factor))
        # Log occasionally
        now = time.ticks_ms()
        if time.ticks_diff(now, _last_power_log) > 2000:
            print(f"Power limited: {total} -> {MAX_POWER} (factor={factor:.2f})")
            _last_power_log = now

    strip.write()


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


def check_interrupt():
    """Check if animation should be interrupted (e.g., power off from HA).
    Also updates brightness dynamically."""
    if MQTT_AVAILABLE and mqtt.connected:
        result = mqtt.should_interrupt()
        update_brightness()  # Update brightness on every check
        return result
    return False


def clear():
    """Turn off all LEDs."""
    for i in range(LED_COUNT):
        strip[i] = (0, 0, 0)
    strip.write()  # No power limiting needed for off


def fill(color):
    """Fill all LEDs with a single color."""
    for i in range(LED_COUNT):
        strip[i] = scale(color)
    safe_write()


# ─── Animations ───────────────────────────────────────────────────────────────


def rainbow_cycle(wait_ms=20, cycles=1):
    """Rainbow that moves along the strip."""
    for _ in range(cycles):
        for j in range(256):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                strip[i] = scale(wheel((i * 256 // LED_COUNT + j) & 255))
            safe_write()
            time.sleep_ms(wait_ms)


def chase(color, wait_ms=50, cycles=3):
    """Theater chase animation."""
    for _ in range(cycles):
        for offset in range(3):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                strip[i] = scale(color) if i % 3 == offset else (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


def chase_christmas(wait_ms=50, cycles=10):
    """Theater chase with alternating red and green."""
    red = (255, 0, 0)
    green = (0, 255, 0)
    for _ in range(cycles):
        for offset in range(3):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                if i % 3 == offset:
                    strip[i] = scale(red if (i // 3) % 2 == 0 else green)
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


def sparkle(color, wait_ms=50, density=0.05, duration_ms=5000):
    """Random twinkling lights."""
    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        if check_interrupt():
            return
        for i in range(LED_COUNT):
            strip[i] = scale(color) if random.random() < density else (0, 0, 0)
        safe_write()
        time.sleep_ms(wait_ms)


def comet(color, tail_length=20, wait_ms=20, cycles=2, count=1):
    """Multiple comets with fading tails."""
    spacing = LED_COUNT // count
    for _ in range(cycles):
        for frame in range(LED_COUNT + tail_length):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                brightness = 0.0
                # Check each comet
                for c in range(count):
                    head = (frame + c * spacing) % (LED_COUNT + tail_length)
                    distance = head - i
                    if 0 <= distance < tail_length:
                        fade = 1.0 - (distance / tail_length)
                        brightness = max(brightness, fade)
                if brightness > 0:
                    strip[i] = scale(tuple(int(v * brightness) for v in color))
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


def candy_cane(wait_ms=100, cycles=50):
    """Red and white stripes that move."""
    stripe_width = 10
    for offset in range(cycles):
        if check_interrupt():
            return
        for i in range(LED_COUNT):
            if ((i + offset) // stripe_width) % 2 == 0:
                strip[i] = scale((255, 0, 0))
            else:
                strip[i] = scale((255, 255, 255))
        safe_write()
        time.sleep_ms(wait_ms)


def twinkle_multi(wait_ms=100, duration_ms=10000):
    """Twinkling multicolor Christmas lights."""
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, green, blue, yellow
    # Initialize with random colors
    led_colors = [random.choice(colors) for _ in range(LED_COUNT)]
    led_brightness = [random.random() for _ in range(LED_COUNT)]

    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        if check_interrupt():
            return
        for i in range(LED_COUNT):
            # Randomly change some LEDs
            if random.random() < 0.02:
                led_colors[i] = random.choice(colors)
                led_brightness[i] = random.random()
            b = led_brightness[i]
            strip[i] = scale(tuple(int(c * b) for c in led_colors[i]))
        safe_write()
        time.sleep_ms(wait_ms)


def wave(color1, color2, wait_ms=50, cycles=3):
    """Smooth wave between two colors."""
    import math
    for _ in range(cycles):
        for offset in range(LED_COUNT):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                ratio = (1 + math.sin((i + offset) * 0.1)) / 2
                r = int(color1[0] * ratio + color2[0] * (1 - ratio))
                g = int(color1[1] * ratio + color2[1] * (1 - ratio))
                b = int(color1[2] * ratio + color2[2] * (1 - ratio))
                strip[i] = scale((r, g, b))
            safe_write()
            time.sleep_ms(wait_ms)


def warm_white(duration_ms=10000):
    """Solid warm white glow."""
    warm = (255, 140, 50)  # Warm white (more orange/candle-like)
    for i in range(LED_COUNT):
        strip[i] = scale(warm)
    safe_write()

    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        if check_interrupt():
            return
        time.sleep_ms(100)


# ─── Main Loop ────────────────────────────────────────────────────────────────


# "auto" cycles through all effects; selecting any other effect locks to it
ANIMATIONS = [
    ("auto", None),  # Special: cycles through all other effects
    ("rainbow_cycle", lambda: rainbow_cycle(20, 2)),
    ("chase_red", lambda: chase((255, 0, 0), 50, 10)),
    ("chase_green", lambda: chase((0, 255, 0), 50, 10)),
    ("chase_christmas", lambda: chase_christmas(50, 20)),
    ("sparkle", lambda: sparkle((255, 255, 255), 30, 0.03, 8000)),
    ("candy_cane", lambda: candy_cane(80, 100)),
    ("twinkle_multi", lambda: twinkle_multi(80, 10000)),
    ("warm_white", lambda: warm_white(10000)),
]


def update_brightness():
    """Update brightness from MQTT state."""
    global _brightness
    if MQTT_AVAILABLE and mqtt.connected:
        _brightness = mqtt.state["brightness"] / 255.0
    else:
        _brightness = BRIGHTNESS


print("Christmas tree starting...")

# Start WiFi and OTA server (non-blocking)
if OTA_AVAILABLE:
    try:
        ota.connect_wifi()
        ota.start_server()
    except Exception as e:
        print(f"OTA setup failed: {e}")
        print("Continuing without OTA...")

# Start MQTT (requires WiFi from OTA)
if MQTT_AVAILABLE:
    try:
        animation_names = [name for name, _ in ANIMATIONS]
        mqtt.setup(animation_names)
        mqtt.state["brightness"] = int(BRIGHTNESS * 255)
    except Exception as e:
        print(f"MQTT setup failed: {e}")

try:
    if OTA_AVAILABLE:
        ota.total_animations = len(ANIMATIONS)

    animation_names = [name for name, _ in ANIMATIONS]
    auto_mode = True  # Start in auto-cycle mode
    auto_index = 1    # Current position in auto cycle (skip index 0 which is "auto")
    locked_index = 0  # Which effect we're locked to (if not auto)

    while True:
        # Check MQTT commands and handle effect changes
        if MQTT_AVAILABLE and mqtt.connected:
            mqtt.check_messages()
            cmd = mqtt.get_pending_command()
            if cmd:
                changes = mqtt.process_command(cmd)
                if changes:
                    print(f"MQTT changes: {changes}")
                    if "effect" in changes:
                        effect_name = changes["effect"]
                        if effect_name == "auto":
                            auto_mode = True
                            print("Switched to auto mode")
                        else:
                            try:
                                locked_index = animation_names.index(effect_name)
                                auto_mode = False
                                print(f"Locked to: {effect_name}")
                            except ValueError:
                                pass

        # Handle power off state
        if MQTT_AVAILABLE and mqtt.connected and not mqtt.state["power"]:
            clear()
            time.sleep_ms(100)
            continue

        # Update brightness before animation
        update_brightness()

        # Determine which animation to run
        if auto_mode:
            animation_index = auto_index
        else:
            animation_index = locked_index

        # Run current animation
        name, anim = ANIMATIONS[animation_index]
        if OTA_AVAILABLE:
            ota.current_animation = name
            ota.animation_index = animation_index
        if MQTT_AVAILABLE and mqtt.connected:
            # In auto mode, report "auto" as the effect; otherwise report the actual effect
            mqtt.state["effect"] = "auto" if auto_mode else name
            mqtt.publish_state()

        if anim:  # Skip if None (the "auto" entry itself)
            anim()

        # Advance auto cycle (only in auto mode)
        if auto_mode:
            auto_index += 1
            if auto_index >= len(ANIMATIONS):
                auto_index = 1  # Skip "auto" at index 0

except KeyboardInterrupt:
    clear()
    print("LEDs off")
