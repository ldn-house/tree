"""Settings from the user's /config.py (USB-installed, never written over
the network), with defaults for anything missing. A missing or broken
config.py must not stop the firmware: the LEDs still run, just offline."""

try:
    import config as _c

    ERROR = None
except BaseException as e:  # SyntaxError, ImportError, anything
    _c = None
    ERROR = repr(e)


def get(name, default=None):
    return getattr(_c, name, default) if _c is not None else default


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
POWER_BUDGET_PERCENT = get("POWER_BUDGET_PERCENT", 13)
GAMMA = get("GAMMA")
FPS = get("FPS", 40)

# Health needed before a freshly installed update is confirmed.
TRIAL_HEALTHY_S = get("TRIAL_HEALTHY_S", 60)
TRIAL_DEADLINE_S = get("TRIAL_DEADLINE_S", 300)
