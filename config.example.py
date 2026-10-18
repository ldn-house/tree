"""
Device settings. Copy to config.py, fill in, and install over USB:
    ./flash.py --config
Every setting is checked against firmware/cfgcheck.py (flash.py runs the same
check first); a config with an invalid setting is not used.
Over Wi-Fi, ./flash.py --ota --config sends it as signed JSON values; the
device tries it and reverts unless it reaches the network. UPDATE_KEY can
only be changed over USB.
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
# GAMMA = None                # e.g. 2.2 for perceptual dimming; must be >= 1
# LED_DITHER = True           # temporal dithering: dim colours keep their hue
# WHITE_BALANCE = None        # e.g. (1.0, 0.85, 0.7): per-channel scale, each 0..1
#   (LED_DITHER and WHITE_BALANCE are validated by cfgcheck.py, part of the
#   recovery layer: a board whose recovery layer predates them ignores both
#   until it is updated over USB, and uses the defaults.)

# Power cap, in amps at the strip's 12 V supply. Every frame is scaled so
# the estimated strip current stays at or below POWER_LIMIT_A.
# POWER_LIMIT_A = 1.5         # amps at 12 V (18 W, ~12% of full white). With
#     a single feed at the data-input end, the first JST-SM connector (~3 A)
#     and voltage drop along ~50 m of thin strand wire are the limits, not
#     the 20 A PSU. Raise only after fused power-injection runs are installed
#     and the far end has been checked for dimming.
# LED_MA_PER_CHANNEL = 8      # mA per colour channel of one pixel at full
#                             # duty (~0.3 W per pixel at full white); > 0. Measure!
# POWER_BUDGET_PERCENT = 13   # deprecated alias: % of full white
