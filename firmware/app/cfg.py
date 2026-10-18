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

# Power cap, in real units at the strip's 12 V supply (not the Pico's 5 V).
# Every frame is scaled so the estimated strip current is at most
# POWER_LIMIT_A.
#   POWER_LIMIT_A: amps at 12 V. Default 1.5 A (18 W, ~12% of full white).
#     With a single feed at the data-input end, the limits are the first
#     JST-SM connector (~3 A) and voltage drop along ~50 m of thin strand
#     wire, not the 20 A PSU. Raise it only after fused power-injection runs
#     are installed and the far end has been checked for dimming.
#   LED_MA_PER_CHANNEL: mA one colour channel of one pixel draws at full
#     duty (255). 12 V WS2811 bullets: ~8 mA, so ~0.3 W per pixel at full
#     white. To be measured.
POWER_LIMIT_A = get("POWER_LIMIT_A", 1.5)
LED_MA_PER_CHANNEL = get("LED_MA_PER_CHANNEL", 8)
# Deprecated: a cap as % of all pixels at full white; overrides POWER_LIMIT_A.
POWER_BUDGET_PERCENT = get("POWER_BUDGET_PERCENT")
GAMMA = get("GAMMA")
# Carry each channel's sub-LSB remainder to the next frame (temporal
# dithering), so dim colours keep their hue instead of collapsing to red.
LED_DITHER = get("LED_DITHER", True)
# Optional per-channel white balance (r, g, b), each 0..1, e.g. (1.0, 0.85, 0.7).
WHITE_BALANCE = get("WHITE_BALANCE")
FPS = get("FPS", 40)

# Health needed before a freshly installed update is confirmed.
TRIAL_HEALTHY_S = get("TRIAL_HEALTHY_S", 60)
TRIAL_DEADLINE_S = get("TRIAL_DEADLINE_S", 300)
