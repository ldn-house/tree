"""
Christmas Tree LED Controller - Pi Pico 2W
MicroPython with PIO-based NeoPixel driver + WiFi OTA updates
"""

from machine import Pin, WDT
from neopixel import NeoPixel
import time
import random
import gc

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

# Hardware watchdog - will reset if not fed for 8 seconds
# This catches any hangs in the main loop or animations
wdt = None  # Initialized after WiFi/OTA setup completes

# Power limiting: max total RGB sum across all LEDs
# Each LED can draw up to 60mA at full white (255,255,255)
# 500 LEDs * 765 (255*3) = 382,500 max
# Cap at ~13% to be safe with power supply
MAX_POWER = int(LED_COUNT * 255 * 3 * 0.13)

strip = NeoPixel(Pin(LED_PIN), LED_COUNT)

# Expose strip to ota module for direct calibration writes
if OTA_AVAILABLE:
    ota.strip = strip
    ota.LED_COUNT = LED_COUNT

# Current brightness level (updated by MQTT)
_brightness = BRIGHTNESS

# Calibration mode - state is stored in ota module, we just handle LED updates here
_last_calibration_state = None  # Track last state (cal_led, binary_bit) to avoid unnecessary updates


def reset_calibration_state():
    """Reset calibration tracking state."""
    global _last_calibration_state
    _last_calibration_state = None


def update_calibration_leds():
    """Write calibration pattern to strip (called from main loop for thread safety)."""
    global _last_calibration_state
    if not OTA_AVAILABLE:
        return

    cal_led = ota.calibration_led
    bit = ota.calibration_binary_bit if cal_led == -2 else 0
    state = (cal_led, bit)

    # Only write when state changes
    if state == _last_calibration_state:
        return
    _last_calibration_state = state

    # Moderate brightness - enough for camera, safe for PSU (GRB order)
    cal_color = (80, 80, 80)

    if cal_led == -1:
        for i in range(LED_COUNT):
            strip[i] = (0, 0, 0)
    elif cal_led == -2:
        for i in range(LED_COUNT):
            if (i >> bit) & 1:
                strip[i] = cal_color
            else:
                strip[i] = (0, 0, 0)
    elif 0 <= cal_led < LED_COUNT:
        for i in range(LED_COUNT):
            strip[i] = (255, 255, 255) if i == cal_led else (0, 0, 0)

    strip.write()


def scale(color):
    """Apply brightness scaling and convert RGB to GRB for WS2811."""
    r, g, b = (int(c * _brightness) for c in color)
    return (g, r, b)  # WS2811 uses GRB order


_last_power_log = 0

def safe_write():
    """Write to strip with automatic power limiting (like HDR)."""
    global _last_power_log

    # Skip if in calibration mode (calibration handles its own writes)
    if OTA_AVAILABLE and ota.calibration_mode:
        return

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


def feed_watchdog():
    """Feed the hardware watchdog to prevent reset."""
    if wdt:
        wdt.feed()


def check_interrupt():
    """Check if animation should be interrupted (e.g., power off from HA, or calibration mode).
    Also updates brightness dynamically and feeds the watchdog."""
    # Feed watchdog on every interrupt check (called frequently during animations)
    feed_watchdog()

    # Check for calibration mode - interrupt immediately
    if OTA_AVAILABLE and ota.calibration_mode:
        return True

    if MQTT_AVAILABLE and mqtt.connected:
        result = mqtt.should_interrupt()
        update_brightness()  # Update brightness on every check
        return result
    return False


def clear(force=False):
    """Turn off all LEDs."""
    # Skip if in calibration mode (unless forced by calibration code itself)
    if not force and OTA_AVAILABLE and ota.calibration_mode:
        return
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
    """Rainbow gradient based on height - moves up/down the tree (3D)."""
    coords = load_coords()

    # Precompute normalized heights for each LED
    led_y = []
    has_3d = any(coords)

    if has_3d:
        y_vals = [c[1] for c in coords if c]
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1

        for i in range(LED_COUNT):
            if coords[i]:
                y = coords[i][1]
                led_y.append((y - y_min) / y_range)  # Normalize to 0-1
            else:
                led_y.append(0)

    for _ in range(cycles):
        for j in range(256):
            if check_interrupt():
                return
            for i in range(LED_COUNT):
                if has_3d and coords[i]:
                    # Color based purely on height (vertical gradient)
                    hue = (led_y[i] + j / 256) % 1.0
                    strip[i] = scale(wheel(int(hue * 255)))
                else:
                    # Fallback to linear
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


def christmas(wait_ms=50, cycles=10):
    """Rotating red and green sectors around the tree (3D)."""
    import math
    coords = load_coords()
    red = (255, 0, 0)
    green = (0, 255, 0)

    has_3d = any(coords)
    led_theta = []

    if has_3d:
        for i in range(LED_COUNT):
            if coords[i]:
                x, y, z = coords[i]
                led_theta.append(math.atan2(z, x))
            else:
                led_theta.append(0)

    num_sectors = 6  # 3 red, 3 green alternating

    for _ in range(cycles):
        for offset in range(60):  # Rotate through 60 steps
            if check_interrupt():
                return
            rotation = (offset / 60) * 2 * math.pi

            for i in range(LED_COUNT):
                if has_3d and coords[i]:
                    theta = led_theta[i] + rotation
                    sector = int(((theta + math.pi) / (2 * math.pi)) * num_sectors) % num_sectors
                    strip[i] = scale(red if sector % 2 == 0 else green)
                else:
                    # Fallback
                    if i % 3 == offset % 3:
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
    """Comets that rise up the tree (3D)."""
    coords = load_coords()
    has_3d = any(coords)

    if has_3d:
        y_vals = [c[1] for c in coords if c]
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1
        tail_size = 0.15  # Tail as fraction of tree height

    for _ in range(cycles):
        for frame in range(100):
            if check_interrupt():
                return

            for i in range(LED_COUNT):
                brightness = 0.0

                if has_3d and coords[i]:
                    y_norm = (coords[i][1] - y_min) / y_range

                    # Multiple comets evenly spaced
                    for c in range(count):
                        head_y = (frame / 100 + c / count) % 1.0
                        distance = head_y - y_norm

                        if 0 <= distance < tail_size:
                            fade = 1.0 - (distance / tail_size)
                            brightness = max(brightness, fade)
                else:
                    # Fallback to linear
                    spacing = LED_COUNT // count
                    for c in range(count):
                        head = (frame * LED_COUNT // 100 + c * spacing) % (LED_COUNT + tail_length)
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
    """Red and white horizontal stripes that move up the tree (3D)."""
    coords = load_coords()
    has_3d = any(coords)

    if has_3d:
        y_vals = [c[1] for c in coords if c]
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1

    num_stripes = 8  # Number of stripe pairs

    for offset in range(cycles):
        if check_interrupt():
            return
        phase = offset / cycles  # 0 to 1

        for i in range(LED_COUNT):
            if has_3d and coords[i]:
                y_norm = (coords[i][1] - y_min) / y_range
                stripe_pos = (y_norm * num_stripes + phase) % 1.0
                if stripe_pos < 0.5:
                    strip[i] = scale((255, 0, 0))  # Red
                else:
                    strip[i] = scale((255, 255, 255))  # White
            else:
                # Fallback to linear
                stripe_width = 10
                if ((i + offset) // stripe_width) % 2 == 0:
                    strip[i] = scale((255, 0, 0))
                else:
                    strip[i] = scale((255, 255, 255))
        safe_write()
        time.sleep_ms(wait_ms)


def twinkle_multi(wait_ms=50, duration_ms=10000):
    """Twinkling multicolor Christmas lights with smooth fading."""
    import math
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 180, 0)]  # Red, green, blue, yellow (warmer)
    # Initialize with random colors and phases
    led_colors = [random.choice(colors) for _ in range(LED_COUNT)]
    led_phase = [random.random() * 2 * math.pi for _ in range(LED_COUNT)]  # Phase in twinkle cycle
    led_speed = [0.03 + random.random() * 0.07 for _ in range(LED_COUNT)]  # Gentle twinkle speeds

    start = time.ticks_ms()
    while time.ticks_diff(time.ticks_ms(), start) < duration_ms:
        if check_interrupt():
            return
        for i in range(LED_COUNT):
            # Advance phase
            led_phase[i] += led_speed[i]
            # Subtle brightness oscillation (0.6 to 1.0 range for gentle effect)
            b = 0.6 + 0.4 * (0.5 + 0.5 * math.sin(led_phase[i]))
            # Randomly change color when LED is at its dimmest point
            if led_phase[i] > 2 * math.pi:
                led_phase[i] -= 2 * math.pi
                if random.random() < 0.3:  # 30% chance to change color each cycle
                    led_colors[i] = random.choice(colors)
                    led_speed[i] = 0.03 + random.random() * 0.07  # New random speed
            strip[i] = scale(tuple(int(c * b) for c in led_colors[i]))
        safe_write()
        time.sleep_ms(wait_ms)


def wave(color1, color2, wait_ms=50, cycles=3):
    """Smooth color wave that moves up the tree (3D)."""
    import math
    coords = load_coords()
    has_3d = any(coords)

    if has_3d:
        y_vals = [c[1] for c in coords if c]
        y_min, y_max = min(y_vals), max(y_vals)
        y_range = y_max - y_min if y_max > y_min else 1

    for _ in range(cycles):
        for offset in range(100):
            if check_interrupt():
                return
            phase = offset / 100 * 2 * math.pi

            for i in range(LED_COUNT):
                if has_3d and coords[i]:
                    y_norm = (coords[i][1] - y_min) / y_range
                    ratio = (1 + math.sin(y_norm * 4 * math.pi + phase)) / 2
                else:
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


# ─── 3D Animations ────────────────────────────────────────────────────────────

# 3D coordinates: list of (x, y, z) tuples, None for missing LEDs
_coords_3d = None


def load_coords():
    """Load 3D coordinates from compact file."""
    global _coords_3d
    if _coords_3d is not None:
        return _coords_3d

    try:
        _coords_3d = []
        with open("coords_compact.txt", "r") as f:
            for line in f:
                line = line.strip()
                if line == "null" or not line:
                    _coords_3d.append(None)
                else:
                    parts = line.split(",")
                    _coords_3d.append((float(parts[0]), float(parts[1]), float(parts[2])))
        print(f"Loaded {sum(1 for c in _coords_3d if c)} 3D coordinates")
    except Exception as e:
        print(f"Could not load 3D coords: {e}")
        _coords_3d = [None] * LED_COUNT

    return _coords_3d


def plane_sweep(color=(0, 255, 0), wait_ms=10, cycles=3):
    """Horizontal plane sweeps up and down the tree."""
    coords = load_coords()
    if not any(coords):
        return

    # Get Y range
    y_vals = [c[1] for c in coords if c]
    y_min, y_max = min(y_vals), max(y_vals)
    y_range = y_max - y_min

    for _ in range(cycles):
        # Sweep up
        for step in range(50):
            if check_interrupt():
                return
            plane_y = y_min + (step / 50) * y_range
            thickness = y_range * 0.03  # 3% of tree height (~2 inches)

            for i in range(LED_COUNT):
                if coords[i]:
                    dist = abs(coords[i][1] - plane_y)
                    if dist < thickness:
                        brightness = 1.0 - (dist / thickness)
                        strip[i] = scale(tuple(int(c * brightness) for c in color))
                    else:
                        strip[i] = (0, 0, 0)
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)

        # Sweep down
        for step in range(50, 0, -1):
            if check_interrupt():
                return
            plane_y = y_min + (step / 50) * y_range
            thickness = y_range * 0.03

            for i in range(LED_COUNT):
                if coords[i]:
                    dist = abs(coords[i][1] - plane_y)
                    if dist < thickness:
                        brightness = 1.0 - (dist / thickness)
                        strip[i] = scale(tuple(int(c * brightness) for c in color))
                    else:
                        strip[i] = (0, 0, 0)
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


def radial_burst(wait_ms=20, cycles=3):
    """Expanding rings from center of tree."""
    coords = load_coords()
    if not any(coords):
        return

    import math

    for _ in range(cycles):
        for radius in range(0, 120, 2):  # Expand outward
            if check_interrupt():
                return
            r_norm = radius / 100.0  # Normalize to ~1.0 max

            for i in range(LED_COUNT):
                if coords[i]:
                    x, y, z = coords[i]
                    # Distance from vertical axis (center of tree)
                    dist = math.sqrt(x * x + z * z)
                    diff = abs(dist - r_norm)

                    if diff < 0.15:  # Ring thickness
                        brightness = 1.0 - (diff / 0.15)
                        # Color based on height
                        hue = y / 2.0  # Assumes y is 0-2 range
                        strip[i] = scale(wheel(int(hue * 255) % 256))
                    else:
                        strip[i] = (0, 0, 0)
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


def spiral_3d(wait_ms=30, cycles=2):
    """Spiral pattern that rotates around the tree."""
    coords = load_coords()
    if not any(coords):
        return

    import math

    for _ in range(cycles):
        for angle_offset in range(0, 360, 5):
            if check_interrupt():
                return
            offset_rad = math.radians(angle_offset)

            for i in range(LED_COUNT):
                if coords[i]:
                    x, y, z = coords[i]
                    # Angle of this LED around the tree
                    led_angle = math.atan2(z, x)
                    # Spiral: angle depends on height
                    target_angle = (y * 3) + offset_rad  # 3 twists up the tree

                    # Angular distance
                    diff = abs(((led_angle - target_angle + math.pi) % (2 * math.pi)) - math.pi)

                    if diff < 0.5:  # Spiral width
                        brightness = 1.0 - (diff / 0.5)
                        # Christmas colors based on which spiral arm
                        if (angle_offset // 60) % 2 == 0:
                            color = (255, 0, 0)  # Red
                        else:
                            color = (0, 255, 0)  # Green
                        strip[i] = scale(tuple(int(c * brightness) for c in color))
                    else:
                        strip[i] = (0, 0, 0)
                else:
                    strip[i] = (0, 0, 0)
            safe_write()
            time.sleep_ms(wait_ms)


# ─── Main Loop ────────────────────────────────────────────────────────────────


# "auto" cycles through all effects; selecting any other effect locks to it
ANIMATIONS = [
    ("auto", None),  # Special: cycles through all other effects
    ("rainbow_cycle", lambda: rainbow_cycle(5, 5)),
    ("christmas", lambda: christmas(30, 20)),
    ("sparkle", lambda: sparkle((255, 255, 255), 80, 0.03, 8000)),
    ("candy_cane", lambda: candy_cane(50, 100)),
    ("twinkle_multi", lambda: twinkle_multi(50, 10000)),
    ("warm_white", lambda: warm_white(10000)),
    ("plane_sweep", lambda: plane_sweep((0, 255, 0), 8, 3)),
    ("radial_burst", lambda: radial_burst(12, 3)),
    ("spiral_3d", lambda: spiral_3d(18, 2)),
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

# Initialize hardware watchdog after all setup is complete
# 8 second timeout - must feed before this or system resets
try:
    wdt = WDT(timeout=8000)
    print("Watchdog enabled (8s timeout)")
except Exception as e:
    print(f"Watchdog init failed: {e}")

try:
    if OTA_AVAILABLE:
        ota.total_animations = len(ANIMATIONS)

    animation_names = [name for name, _ in ANIMATIONS]
    auto_mode = True  # Start in auto-cycle mode
    auto_index = 1    # Current position in auto cycle (skip index 0 which is "auto")
    locked_index = 0  # Which effect we're locked to (if not auto)
    was_calibrating = False  # Track calibration state transitions
    was_off = False  # Track power state to only clear once

    # Periodic maintenance
    _last_gc = 0
    _gc_interval = 30000  # GC every 30 seconds

    _cal_debug_counter = 0
    while True:
        # Feed watchdog at start of each loop iteration
        feed_watchdog()

        # Periodic garbage collection to prevent memory fragmentation
        now = time.ticks_ms()
        if time.ticks_diff(now, _last_gc) > _gc_interval:
            _last_gc = now
            gc.collect()

        # Check WiFi and reconnect if needed
        if OTA_AVAILABLE:
            ota.check_wifi()

        # Handle calibration mode - skip normal animation processing
        if OTA_AVAILABLE and ota.calibration_mode:
            was_calibrating = True
            # Update LEDs based on calibration commands
            update_calibration_leds()

            # Minimal loop - just refresh LEDs, no MQTT during calibration
            time.sleep_ms(100)  # 100ms between refreshes
            continue

        # Reset calibration state when exiting calibration mode
        if was_calibrating:
            reset_calibration_state()
            was_calibrating = False
            clear()
            print("Exited calibration mode, resuming animations")

        # Check MQTT commands and handle effect changes (also handles reconnection)
        if MQTT_AVAILABLE:
            mqtt.check_messages()  # This handles reconnection when disconnected
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

        # Handle power off state - only clear once, then just wait
        if MQTT_AVAILABLE and mqtt.connected and not mqtt.state["power"]:
            if not was_off:
                clear()
                was_off = True
                print("Power off - LEDs cleared")
            time.sleep_ms(100)
            continue
        else:
            was_off = False

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
