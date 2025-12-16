"""
MQTT integration for Home Assistant.
Exposes the tree as a light entity with brightness and effects.
"""

import json
import time

try:
    from umqtt.simple import MQTTClient
except ImportError:
    MQTTClient = None

import config

# MQTT Topics
TOPIC_PREFIX = "homeassistant/light/xmas_tree"
DISCOVERY_TOPIC = f"{TOPIC_PREFIX}/config"
STATE_TOPIC = "xmas_tree/state"
COMMAND_TOPIC = "xmas_tree/set"
ATTRIBUTES_TOPIC = "xmas_tree/attributes"

# State (modified by main.py and commands)
state = {
    "power": True,
    "brightness": 255,  # 0-255 for HA
    "effect": "rainbow_cycle",
}

# Available effects (populated from main.py's ANIMATIONS)
effects_list = []

# Command queue for main loop to process
pending_commands = []

# MQTT client instance
client = None
connected = False

# Reconnection state
_last_reconnect_attempt = 0
_reconnect_backoff = 5000  # Start with 5 seconds
_max_reconnect_backoff = 60000  # Max 60 seconds
_consecutive_failures = 0


def _on_message(topic, msg):
    """Handle incoming MQTT commands."""
    global pending_commands
    try:
        topic_str = topic.decode() if isinstance(topic, bytes) else topic
        payload = json.loads(msg.decode() if isinstance(msg, bytes) else msg)
        print(f"MQTT received: {payload}")
        pending_commands.append(payload)
    except Exception as e:
        print(f"MQTT message error: {e}")


def connect():
    """Connect to MQTT broker and subscribe to command topic."""
    global client, connected, _reconnect_backoff, _consecutive_failures

    if MQTTClient is None:
        print("MQTT: umqtt.simple not available")
        return False

    try:
        client_id = f"xmas_tree_{config.HOSTNAME}"
        client = MQTTClient(
            client_id,
            config.MQTT_BROKER,
            port=config.MQTT_PORT,
            user=config.MQTT_USER,
            password=config.MQTT_PASSWORD,
            keepalive=60,
        )
        client.set_callback(_on_message)
        client.connect()
        print(f"MQTT subscribing to: {COMMAND_TOPIC}")
        client.subscribe(COMMAND_TOPIC.encode())
        connected = True
        # Reset backoff on successful connection
        _reconnect_backoff = 5000
        _consecutive_failures = 0
        print(f"MQTT connected to {config.MQTT_BROKER}")
        return True
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        connected = False
        _consecutive_failures += 1
        return False


def reconnect():
    """Attempt to reconnect to MQTT broker with exponential backoff."""
    global _last_reconnect_attempt, _reconnect_backoff, connected, client

    if connected:
        return True

    now = time.ticks_ms()
    if time.ticks_diff(now, _last_reconnect_attempt) < _reconnect_backoff:
        return False  # Not time yet

    _last_reconnect_attempt = now
    print(f"MQTT reconnecting (backoff: {_reconnect_backoff}ms)...")

    # Clean up old client
    if client:
        try:
            client.disconnect()
        except:
            pass
        client = None

    if connect():
        # Re-publish discovery, state, and attributes after reconnection
        publish_discovery()
        publish_state()
        publish_attributes()
        return True
    else:
        # Increase backoff for next attempt (exponential with cap)
        _reconnect_backoff = min(_reconnect_backoff * 2, _max_reconnect_backoff)
        return False


def publish_discovery():
    """Publish Home Assistant MQTT discovery config."""
    if not connected or client is None:
        return

    discovery_payload = {
        "name": "Christmas Tree",
        "unique_id": "xmas_tree_pico",
        "command_topic": COMMAND_TOPIC,
        "state_topic": STATE_TOPIC,
        "json_attributes_topic": ATTRIBUTES_TOPIC,
        "schema": "json",
        "brightness": True,
        "brightness_scale": 255,
        "effect": True,
        "effect_list": effects_list,
        "device": {
            "identifiers": ["xmas_tree_pico"],
            "name": "Christmas Tree",
            "model": "Pi Pico 2W",
            "manufacturer": "DIY",
        },
    }

    try:
        client.publish(
            DISCOVERY_TOPIC.encode(),
            json.dumps(discovery_payload).encode(),
            retain=True,
        )
        print("MQTT discovery published")
    except Exception as e:
        print(f"MQTT discovery error: {e}")


def publish_state():
    """Publish current state to Home Assistant."""
    if not connected or client is None:
        return

    state_payload = {
        "state": "ON" if state["power"] else "OFF",
        "brightness": state["brightness"],
        "effect": state["effect"],
    }

    try:
        client.publish(
            STATE_TOPIC.encode(),
            json.dumps(state_payload).encode(),
            retain=True,
        )
    except Exception as e:
        print(f"MQTT state publish error: {e}")


def publish_attributes():
    """Publish device attributes (IP, uptime, memory, WiFi signal, version)."""
    if not connected or client is None:
        return

    import gc

    attrs = {}

    # Get info from OTA module
    try:
        import ota
        attrs["ip_address"] = ota._ip_address or "unknown"
        if ota._start_time:
            attrs["uptime_seconds"] = int(time.time() - ota._start_time)
        attrs["current_animation"] = ota.current_animation or "unknown"
    except ImportError:
        pass

    # Version info
    try:
        import version
        attrs["firmware_version"] = version.GIT_COMMIT
        attrs["firmware_branch"] = version.GIT_BRANCH
        attrs["firmware_build_time"] = version.BUILD_TIME
    except (ImportError, AttributeError):
        attrs["firmware_version"] = "unknown"

    # Memory info
    gc.collect()
    attrs["free_memory_bytes"] = gc.mem_free()

    # WiFi signal strength (RSSI)
    try:
        import network
        wlan = network.WLAN(network.STA_IF)
        if wlan.isconnected():
            rssi = wlan.status("rssi")
            attrs["wifi_rssi"] = rssi
            # Signal quality as percentage (rough approximation)
            # RSSI typically ranges from -30 (excellent) to -90 (unusable)
            quality = max(0, min(100, 2 * (rssi + 100)))
            attrs["wifi_signal_percent"] = quality
    except Exception:
        pass

    try:
        client.publish(
            ATTRIBUTES_TOPIC.encode(),
            json.dumps(attrs).encode(),
            retain=True,
        )
    except Exception as e:
        print(f"MQTT attributes publish error: {e}")


_last_ping = 0
_ping_interval = 30000  # Ping every 30 seconds to keep connection alive

_last_attributes = 0
_attributes_interval = 60000  # Publish attributes every 60 seconds


def check_messages():
    """Non-blocking check for incoming messages. Call this frequently."""
    global connected, _last_ping, _last_attributes

    # Try to reconnect if disconnected
    if not connected or client is None:
        reconnect()
        return

    try:
        # Set socket to non-blocking for check_msg
        client.sock.setblocking(False)
        try:
            client.check_msg()
        finally:
            client.sock.setblocking(True)

        now = time.ticks_ms()

        # Periodic ping to keep connection alive
        if time.ticks_diff(now, _last_ping) > _ping_interval:
            _last_ping = now
            try:
                client.ping()
            except Exception as e:
                print(f"MQTT ping failed: {e}")
                connected = False

        # Periodic attributes publishing
        if time.ticks_diff(now, _last_attributes) > _attributes_interval:
            _last_attributes = now
            publish_attributes()

    except OSError:
        pass  # No message waiting (EAGAIN)
    except Exception as e:
        print(f"MQTT check error: {e}")
        connected = False


def process_command(cmd):
    """
    Process a command from Home Assistant.
    Returns dict of changes to apply, or None.
    """
    changes = {}

    if "state" in cmd:
        new_power = cmd["state"] == "ON"
        if new_power != state["power"]:
            state["power"] = new_power
            changes["power"] = new_power

    if "brightness" in cmd:
        new_brightness = int(cmd["brightness"])
        if new_brightness != state["brightness"]:
            state["brightness"] = new_brightness
            changes["brightness"] = new_brightness

    if "effect" in cmd:
        new_effect = cmd["effect"]
        if new_effect != state["effect"] and new_effect in effects_list:
            state["effect"] = new_effect
            changes["effect"] = new_effect

    if changes:
        publish_state()

    return changes if changes else None


def get_pending_command():
    """Get next pending command, or None."""
    global pending_commands
    if pending_commands:
        return pending_commands.pop(0)
    return None


def setup(animation_names):
    """Initialize MQTT with list of available animations."""
    global effects_list
    effects_list = animation_names

    if connect():
        publish_discovery()
        publish_state()
        publish_attributes()
        return True
    return False


def should_interrupt():
    """
    Check for MQTT commands and return True if animation should stop.
    Call this periodically in animation loops for responsive control.
    Only processes power commands here; effect commands are handled by main loop.
    """
    check_messages()

    # Only process power commands here, leave effect commands for main loop
    for cmd in list(pending_commands):
        if "state" in cmd and "effect" not in cmd:
            # Power-only command, process it
            pending_commands.remove(cmd)
            changes = process_command(cmd)
            if changes and "power" in changes and not state["power"]:
                return True

    # Check if power is off
    if not state["power"]:
        return True

    # Check if there's an effect command waiting (don't process, just signal)
    for cmd in pending_commands:
        if "effect" in cmd:
            return True  # Interrupt so main loop can handle effect change

    return False


def get_requested_effect():
    """Get the currently requested effect from HA, or None."""
    return state.get("effect")
