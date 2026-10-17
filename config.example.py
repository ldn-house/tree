"""
Device settings. Copy to config.py, fill in, and install over USB:
    ./flash.py --config
config.py is never sent or changed over the network.
"""

WIFI_SSID = "YOUR_WIFI_SSID"
WIFI_PASSWORD = "YOUR_WIFI_PASSWORD"

# Hostname; the device answers as <HOSTNAME>.local via the rp2 lwIP mDNS.
HOSTNAME = "tree"

# Control/OTA HTTP server.
HTTP_PORT = 8080

# Shared secret for every mutating HTTP endpoint (updates, reboot, state,
# calibration). Send it as "Authorization: Bearer <token>". Without it those
# endpoints refuse everything. Generate one with:
#     python3 -c "import secrets; print(secrets.token_urlsafe(24))"
HTTP_TOKEN = "CHANGE-ME"

# MQTT / Home Assistant (leave MQTT_BROKER = None to run without MQTT).
# Use an IP address or a unicast DNS name (.local names are not resolved).
MQTT_BROKER = "192.168.2.x"
MQTT_PORT = 1883
MQTT_USER = "your_ha_username"
MQTT_PASSWORD = "your_ha_password"

# Firmware signing key (HMAC-SHA256), the same string as the repository's
# FIRMWARE_SIGNING_KEY Actions secret. When set, the device only installs
# bundles signed with it, and can update itself from GitHub Releases.
# UPDATE_KEY = "..."
# UPDATE_CHANNEL = "main"   # "main" (every push to main) or "stable" (tags)

# Hardware. COLOR_ORDER is the order the strip expects on the wire.
# LED_COUNT = 500
# LED_PIN = 0
# COLOR_ORDER = "RGB"
# LED_TIMING_NS = (400, 850, 800, 450)  # T0H, T0L, T1H, T1L in ns (800 kHz);
#     MicroPython neopixel's timing. The BSS138 level shifter's slow rising
#     edge means T0H shouldn't go much below 400 ns.
# GAMMA = None                # e.g. 2.2 for perceptual dimming

# Power, at the strip's 12 V supply (PSU: 12 V 20 A). The firmware scales
# every frame so the estimated strip current stays within POWER_BUDGET_MA.
# LED_MA_PER_CHANNEL = 8      # mA per colour channel of one pixel at full
#                             # duty; ~0.3 W per pixel at full white. Measure!
# POWER_BUDGET_MA = 1560      # mA at 12 V. 1560 (18.7 W) equals the old 13%
#                             # cap; raise it only after measuring the draw.
