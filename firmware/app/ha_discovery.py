"""Home Assistant discovery, loaded for a pass and then released."""

import gc
import hashlib
import json
import time

from app import hw
from app.log import log

# Retained topics the old single-file firmware published that nothing reads
# any more (its light attributes, with last year's IP and RSSI). Cleared with
# an empty retained message on each discovery pass: harmless when already gone.
LEGACY_RETAINED = ("attributes",)


def _title(name):
    name = name.replace("_", " ")
    return name[:1].upper() + name[1:]  # no str.capitalize() in MicroPython


def _param_template(eff, pname):
    return ("{%% set p = value_json.params %%}"
            "{{ p[%r][%r] if %r in p and %r in p[%r] else 'None' }}" % (eff, pname, eff, pname, eff))


def _step(lo, hi):
    if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool):
        return 1
    span = hi - lo
    for step in (0.001, 0.01, 0.1, 1, 10):
        if span / step <= 200:
            return step
    return 100


def _device(self):
    rt = self.rt
    dev = {
        "identifiers": ["xmas_tree_pico"],
        "name": "Christmas Tree",
        "manufacturer": "DIY",
        "model": "Pi Pico 2 W, %d x WS2811" % rt.n,
        "sw_version": rt.version,
    }
    if rt.wifi.ip:
        dev["configuration_url"] = "http://%s:%d/" % (rt.wifi.ip, rt.cfg.HTTP_PORT)
    return dev

def _entities(self):
    """Yield (component, object_id, config) for every entity."""
    p = self.p
    names = self.rt.fx.names()
    diag = "diagnostic"
    yield "light", None, {
        "name": None,
        "unique_id": "xmas_tree_pico",
        "schema": "json",
        "command_topic": self.t_set,
        "state_topic": self.t_state,
        "supported_color_modes": ["brightness"],
        "brightness_scale": 255,
        "effect": True,
        "effect_list": ["auto"] + names,
        # The firmware before this one pointed the light's attributes at
        # xmas_tree/attributes, and HA keeps a key a newer config omits;
        # name the topic (and how to parse it) explicitly.
        "json_attributes_topic": self.t_effect_attr,
        "json_attributes_template": "{{ value }}",
    }
    yield "number", "speed", {
        "name": "Speed",
        "command_topic": self.t_cmd + "speed",
        "state_topic": self.t_settings,
        "value_template": "{{ value_json.speed }}",
        "min": 0.1,
        "max": 5.0,
        "step": 0.1,
        "mode": "slider",
        "icon": "mdi:speedometer",
        "entity_category": "config",
    }
    yield "number", "brightness_floor", {
        "name": "Brightness floor",
        "command_topic": self.t_cmd + "brightness_floor",
        "state_topic": self.t_settings,
        "value_template": "{{ value_json.floor }}",
        "min": 0,
        "max": 25,
        "step": 0.5,
        "unit_of_measurement": "%",
        "mode": "slider",
        "icon": "mdi:brightness-percent",
        "entity_category": "config",
    }
    yield "number", "colour_accuracy", {
        # u'v' drift allowed at the bottom of the slider; 0 is off.
        "name": "Colour accuracy",
        "command_topic": self.t_cmd + "colour_accuracy",
        "state_topic": self.t_settings,
        "value_template": "{{ value_json.colour_accuracy }}",
        "min": 0,
        "max": 0.05,
        "step": 0.001,
        "mode": "box",
        "icon": "mdi:palette",
        "entity_category": "config",
    }
    yield "number", "auto_cycle", {
        "name": "Auto-cycle duration",
        "command_topic": self.t_cmd + "auto_cycle",
        "state_topic": self.t_settings,
        "value_template": "{{ value_json.auto_s }}",
        "min": 0,
        "max": 600,
        "step": 5,
        "unit_of_measurement": "s",
        "mode": "box",
        "icon": "mdi:timer-sync-outline",
        "entity_category": "config",
    }
    for eff in names:
        for pname, spec in self.rt.fx.meta(eff)["params"].items():
            yield "number", "param_%s_%s" % (eff, pname), {
                "name": "%s %s" % (_title(eff), pname.replace("_", " ")),
                "command_topic": "%s%s/%s/%s" % (self.t_cmd, "param", eff, pname),
                "state_topic": self.t_settings,
                # "None" (HA's reset payload: unknown) rather than an error
                # when a settings payload lacks the param, e.g. the
                # previous firmware's retained one right after an update.
                "value_template": _param_template(eff, pname),
                "min": spec[1],
                "max": spec[2],
                "step": _step(spec[1], spec[2]),
                "mode": "slider",
                "entity_category": "config",
            }
    yield "sensor", "effect_running", {
        "name": "Running effect",
        "state_topic": self.t_effect,
        # Explicit: this entity used to read {{ value_json.effect }} from
        # diag, and HA keeps a template a newer config omits, failing on
        # every plain-text payload until the integration is reloaded.
        "value_template": "{{ value }}",
        "json_attributes_topic": self.t_effect_attr,
        "json_attributes_template": "{{ value }}",
        "icon": "mdi:animation-play",
    }
    sensors = (
        ("rssi", "Wi-Fi signal", {"device_class": "signal_strength", "unit_of_measurement": "dBm", "state_class": "measurement"}),
        ("heap", "Free memory", {"device_class": "data_size", "unit_of_measurement": "B", "state_class": "measurement"}),
        ("uptime", "Uptime", {"device_class": "duration", "unit_of_measurement": "s", "state_class": "total_increasing"}),
        ("fps", "Frame rate", {"unit_of_measurement": "fps", "state_class": "measurement", "icon": "mdi:filmstrip"}),
        ("cap_pct", "Power cap output", {"unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:flash-triangle"}),
        ("floor", "Brightness floor in effect", {"unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:brightness-percent", "suggested_display_precision": 1}),
        ("floor_colour", "Colour floor", {"unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:palette", "suggested_display_precision": 1}),
        ("led_amps", "LED current (estimated, 12 V)", {"device_class": "current", "unit_of_measurement": "A", "state_class": "measurement", "suggested_display_precision": 2}),
        ("reset", "Last reset cause", {"icon": "mdi:restart-alert"}),
        ("boot", "Boot status", {"icon": "mdi:shield-check"}),
        ("recovery", "Recovery layer", {"icon": "mdi:lifebuoy"}),
        ("ip", "IP address", {"icon": "mdi:ip-network"}),
        ("error", "Last effect error", {"icon": "mdi:alert-circle-outline"}),
    )
    for key, name, extra in sensors:
        cfg = {
            "name": name,
            "state_topic": self.t_diag,
            "value_template": "{{ value_json.%s }}" % key,
            "entity_category": diag,
        }
        cfg.update(extra)
        yield "sensor", key, cfg
    yield "binary_sensor", "power_capped", {
        "name": "Power cap engaged",
        "state_topic": self.t_diag,
        "value_template": "{{ 'ON' if value_json.capped else 'OFF' }}",
        "icon": "mdi:flash-alert",
        "entity_category": diag,
    }
    yield "binary_sensor", "calibrating", {
        "name": "Calibrating",
        "state_topic": self.t_cal_state,
        "value_template": "{{ 'ON' if value_json.state != 'idle' else 'OFF' }}",
        "icon": "mdi:camera-metering-matrix",
        "entity_category": diag,
    }
    yield "button", "reboot", {
        "name": "Reboot",
        "command_topic": self.t_cmd + "reboot",
        "payload_press": "PRESS",
        "device_class": "restart",
        "entity_category": "config",
    }
    yield "button", "calibrate_play", {
        "name": "Play calibration sequence",
        "command_topic": self.t_cal_set,
        "payload_press": '{"action":"play","sequence":"default","loops":0,"delay_ms":0,"ttl_s":300}',
        "icon": "mdi:play-box-outline",
    }
    yield "button", "calibrate_stop", {
        "name": "Stop calibration",
        "command_topic": self.t_cal_set,
        "payload_press": '{"action":"unlock"}',
        "icon": "mdi:stop-circle-outline",
    }
    upd = {
        "name": "Firmware",
        "state_topic": self.t_update,
        "device_class": "firmware",
        "entity_category": "config",
    }
    if self.rt.updates.self_update_enabled():
        upd["command_topic"] = self.t_cmd + "install"
        upd["payload_install"] = "install"
    yield "update", "firmware", upd

def _config_topic(self, comp, obj):
    if obj is None:
        return "%s/%s/xmas_tree/config" % (self.dp, comp)
    return "%s/%s/xmas_tree/%s/config" % (self.dp, comp, obj)


def _configs(self):
    """(topic, JSON config, object id) for every entity."""
    dev = _device(self)
    origin = {"name": "tree firmware", "sw_version": self.rt.version, "support_url": "https://github.com/ldn-house/tree"}
    for comp, obj, cfg in _entities(self):
        self.hb = time.ticks_ms()
        if obj is not None:
            cfg["unique_id"] = "xmas_tree_" + obj
        cfg["availability_topic"] = self.t_avail
        cfg["device"] = dev
        cfg["origin"] = origin
        yield _config_topic(self, comp, obj), json.dumps(cfg), obj


def _digest(self):
    h = hashlib.sha256()
    for topic, js, _obj in _configs(self):
        h.update(topic)
        h.update(js)
    return h.digest()


async def publish(self, force=False):
    """Discovery configs are retained, so they only need publishing when
    they change (a new firmware or effect set), on the first connect,
    or when Home Assistant asks (its birth message). Re-publishing them
    on every reconnect made HA re-initialise every entity each time.
    A pass only counts once every publish in it went out: if the session
    drops part-way, nothing is recorded and the next connect publishes
    it all again (review R3-M3). Returns True when a pass completed."""
    digest = _digest(self)
    if not force and digest == self.discovered:
        return False
    self.discovered = None  # until this pass completes
    # Current settings first, so new param entities never start from
    # the previous firmware's retained payload (which lacks their param).
    if not await self.mqtt.publish(self.t_settings, json.dumps(self.settings_payload()), retain=True):
        log("ha: discovery interrupted; will retry on reconnect")
        return False
    for name in LEGACY_RETAINED:
        if not await self.mqtt.publish(self.p + "/" + name, b"", retain=True):
            log("ha: discovery interrupted; will retry on reconnect")
            return False
    params = []
    n = 0
    for topic, js, obj in _configs(self):
        n += 1
        if obj and obj.startswith("param_"):
            params.append(obj)
        if not await self.mqtt.publish(topic, js, retain=True):
            log("ha: discovery interrupted after %d entities; will retry on reconnect" % (n - 1))
            return False
    # Remove entities for params that no longer exist.
    for obj in self.rt.state.get("ha_params"):
        if obj not in params:
            if not await self.mqtt.publish(_config_topic(self, "number", obj), b"", retain=True):
                log("ha: discovery cleanup interrupted; will retry on reconnect")
                return False
    self.rt.state.set_ha_params(params)
    self.discovered = digest
    self.discoveries += 1
    gc.collect()
    log("ha: discovery published (%d entities, %d bytes free after collect)" % (n, hw.mem_free()))
    return True
