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
# POWER_BUDGET_PERCENT = 13   # per-frame cap, % of every LED at full white
# GAMMA = None                # e.g. 2.2 for perceptual dimming
