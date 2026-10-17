"""Settings from the user's /config.py, with defaults for anything missing.

Loaded through bootmgr.load_config(): if config.py is missing, broken or
lacks the required keys, the last config that did load (config.good.json)
is used instead, so the board stays on the network and can be repaired over
Wi-Fi (PUT /config). A broken config never stops the firmware.
"""

import bootmgr

_values, SOURCE, ERROR = bootmgr.load_config()


def get(name, default=None):
    return _values.get(name, default)


WIFI_SSID = get("WIFI_SSID")
WIFI_PASSWORD = get("WIFI_PASSWORD")
HOSTNAME = get("HOSTNAME", "tree")
HTTP_PORT = get("HTTP_PORT", get("OTA_PORT", 8080))
# Shared secret for every mutating HTTP endpoint. Without one, those
# endpoints refuse all requests (fail closed).
HTTP_TOKEN = get("HTTP_TOKEN")

MQTT_BROKER = get("MQTT_BROKER")
MQTT_PORT = get("MQTT_PORT", 1883)
MQTT_USER = get("MQTT_USER")
MQTT_PASSWORD = get("MQTT_PASSWORD")
MQTT_PREFIX = get("MQTT_PREFIX", "xmas_tree")
MQTT_KEEPALIVE = get("MQTT_KEEPALIVE", 30)
HA_PREFIX = get("HA_DISCOVERY_PREFIX", "homeassistant")

# HMAC-SHA256 key for signed firmware bundles. When set, every bundle (LAN
# push or GitHub) must be signed with it; self-update from GitHub needs it.
UPDATE_KEY = get("UPDATE_KEY")
UPDATE_CHANNEL = get("UPDATE_CHANNEL", "main")
UPDATE_REPO = get("UPDATE_REPO", "ldn-house/tree")
UPDATE_URL = get("UPDATE_URL")  # template with {channel} and {asset}
UPDATE_CHECK_HOURS = get("UPDATE_CHECK_HOURS", 6)

LED_COUNT = get("LED_COUNT", 500)
LED_PIN = get("LED_PIN", 0)
# Order the strip expects on the wire. The WS2811 ICs on this tree take RGB
# (review M8: the old firmware's two swaps cancelled out to RGB).
COLOR_ORDER = get("COLOR_ORDER", "RGB")
# Data waveform in ns: (T0H, T0L, T1H, T1L), 800 kHz. The default is
# MicroPython neopixel's timing; see app/driver.py before changing it.
LED_TIMING_NS = get("LED_TIMING_NS")

# Power model, at the strip's 12 V supply (not the Pico's 5 V):
#   LED_MA_PER_CHANNEL: current one colour channel of one pixel draws at full
#     duty (255). 12 V WS2811 bullets: ~8 mA, so ~24 mA (~0.3 W) per pixel at
#     full white. To be measured on the tree.
#   POWER_BUDGET_MA: the most the strip may draw per frame; the output stage
#     scales the frame down to fit. The default 1560 mA (18.7 W) is exactly
#     the old firmware's 13% cap, kept because higher settings browned out
#     the PSU before. The PSU is 12 V 20 A, so this can be raised once the
#     real draw is measured, e.g. 5000 mA = 60 W.
LED_MA_PER_CHANNEL = get("LED_MA_PER_CHANNEL", 8)
POWER_BUDGET_MA = get("POWER_BUDGET_MA", 1560)
# Legacy: a cap as % of all pixels at full white overrides the above.
POWER_BUDGET_PERCENT = get("POWER_BUDGET_PERCENT")
GAMMA = get("GAMMA")
FPS = get("FPS", 40)

# Health needed before a freshly installed update is confirmed.
TRIAL_HEALTHY_S = get("TRIAL_HEALTHY_S", 60)
TRIAL_DEADLINE_S = get("TRIAL_DEADLINE_S", 300)
