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
    global client, connected

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
        print(f"MQTT connected to {config.MQTT_BROKER}")
        return True
    except Exception as e:
        print(f"MQTT connection failed: {e}")
        connected = False
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


def check_messages():
    """Non-blocking check for incoming messages. Call this frequently."""
    global connected
    if not connected or client is None:
        return

    try:
        # Set socket to non-blocking for check_msg
        client.sock.setblocking(False)
        try:
            client.check_msg()
        finally:
            client.sock.setblocking(True)
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
