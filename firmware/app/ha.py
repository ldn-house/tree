"""Home Assistant over MQTT: discovery, commands and published state.

Everything is grouped under one HA device (identifier "xmas_tree_pico", the
same as the old firmware, so the existing light entity and its automations
carry over). See README.md, "Home Assistant", for the entity list.

Topics (prefix cfg.MQTT_PREFIX, default "xmas_tree"):

    <p>/availability         online/offline (retained; last will = offline)
    <p>/state, <p>/set       the light (JSON schema), as before
    <p>/settings             speed, auto-cycle duration, effect params (JSON)
    <p>/diag                 diagnostics (JSON, every 30 s)
    <p>/effect_running       what the strip shows: effect name, "off" or
                             "calibration" (retained, on change, <= 1/s)
    <p>/effect_running/attributes   {"mode", "selected", "next_switch_s"}
    <p>/update               firmware update entity state (JSON)
    <p>/cmd/...              number/button/update commands
    <p>/calibrate/set|state  calibration (docs/calibration-protocol.md, 7)

The device's persisted state is the truth. Retained messages on command
topics are ignored, so a stale retained "OFF" (or anything else) never
overrides what the tree restored from flash at boot.
"""

import gc
import hashlib
import json
import time

import asyncio

from app import hw, trace
from app.log import log

DIAG_EVERY_MS = 30000
EFFECT_MIN_MS = 1000
REDISCOVER_DELAY_MS = 3000
ONLINE_AGAIN_MS = 1500  # re-assert "online" this long after connecting
RETAINED_WAIT_MS = 1000  # for the broker to replay our retained availability


def _title(name):
    name = name.replace("_", " ")
    return name[:1].upper() + name[1:]  # no str.capitalize() in MicroPython


def _step(lo, hi):
    if isinstance(lo, int) and isinstance(hi, int) and not isinstance(lo, bool):
        return 1
    span = hi - lo
    for step in (0.001, 0.01, 0.1, 1, 10):
        if span / step <= 200:
            return step
    return 100


class HA:
    def __init__(self, rt, mqtt):
        self.rt = rt
        self.mqtt = mqtt
        p = rt.cfg.MQTT_PREFIX
        self.p = p
        self.dp = rt.cfg.HA_PREFIX
        self.t_avail = p + "/availability"
        self.t_state = p + "/state"
        self.t_set = p + "/set"
        self.t_settings = p + "/settings"
        self.t_diag = p + "/diag"
        # What the strip is showing, published on every change (diag is only
        # every 30 s and auto mode switches effects every 8-20 s).
        self.t_effect = p + "/effect_running"
        self.t_effect_attr = p + "/effect_running/attributes"
        self.t_update = p + "/update"
        self.t_cmd = p + "/cmd/"
        self.t_cal_set = p + "/calibrate/set"
        self.t_cal_state = p + "/calibrate/state"
        self.t_ha_status = self.dp + "/status"
        mqtt.will = (self.t_avail, "offline", True)
        # Our own availability too: if the broker publishes the will of an
        # older session over this one's "online", put "online" back.
        mqtt.subs = [self.t_set, self.t_cmd + "#", self.t_cal_set, self.t_ha_status, self.t_avail]
        mqtt.on_connect = self.on_connect
        mqtt.on_message = self.on_message
        self.dirty = {"light": False, "settings": False, "update": False, "calib": False, "diag": False}
        self.effect_shown = None  # (label, mode) last published
        self.effect_at = None  # ticks of that publish
        self.ready = False  # discovery published for this session
        self.rediscover_at = None
        self.discovered = None  # digest of the discovery configs last published
        self.discoveries = 0
        self.online_at = None  # when to (re-)publish "online"
        self.avail_retained = False  # the broker replayed a retained availability
        self.offline_seen = 0
        self.last_diag = 0
        self.last_cal = 0
        self.ignored_retained = 0
        self.bad_commands = 0
        self.hb = time.ticks_ms()
        rt.state.listeners.append(self._on_state)

    # -- state tracking --

    def _on_state(self, changed, source):
        if "power" in changed or "brightness" in changed or "effect" in changed:
            self.dirty["light"] = True
        if "speed" in changed or "auto_s" in changed or "params" in changed:
            self.dirty["settings"] = True

    def mark(self, what):
        self.dirty[what] = True

    # -- incoming --

    def on_message(self, topic, payload, retained):
        if topic == self.t_avail:
            if retained:
                self.avail_retained = True
            if payload == b"offline" and self.mqtt.connected:
                self.offline_seen += 1
                log("ha: availability says offline while connected; re-publishing online")
                self.online_at = time.ticks_ms()
            return
        if topic == self.t_ha_status:
            if payload == b"online":
                self.rediscover_at = time.ticks_add(time.ticks_ms(), REDISCOVER_DELAY_MS)
            return
        if retained:
            # A retained command is stale by definition; persisted state wins.
            self.ignored_retained += 1
            log("ha: ignored retained message on " + topic)
            return
        try:
            if topic == self.t_set:
                self._light(payload)
            elif topic == self.t_cal_set:
                self._calibrate(payload)
            elif topic.startswith(self.t_cmd):
                self._command(topic[len(self.t_cmd) :], payload)
        except ValueError as e:
            self.bad_commands += 1
            log("ha: bad command on %s: %s" % (topic, e))
            self.dirty["light"] = True
            self.dirty["settings"] = True

    def _light(self, payload):
        try:
            cmd = json.loads(payload)
        except ValueError:
            raise ValueError("not JSON")
        if not isinstance(cmd, dict):
            raise ValueError("expected a JSON object")
        changes = {}
        if "state" in cmd:
            if cmd["state"] not in ("ON", "OFF"):
                raise ValueError("state must be ON or OFF")
            changes["power"] = cmd["state"] == "ON"
        if "brightness" in cmd:
            changes["brightness"] = cmd["brightness"]
        if "effect" in cmd:
            changes["effect"] = cmd["effect"]
        self.dirty["light"] = True  # always echo the resulting state
        if changes.get("power") is False:
            self.rt.calib.unlock("power off from Home Assistant")
        self.rt.apply(changes, "mqtt")

    def _calibrate(self, payload):
        try:
            cmd = json.loads(payload)
        except ValueError:
            raise ValueError("not JSON")
        if not isinstance(cmd, dict):
            raise ValueError("expected a JSON object")
        err = self.rt.calib.mqtt_command(cmd)
        if err:
            raise ValueError(err)

    def _command(self, name, payload):
        text = payload.decode().strip()
        if name == "speed":
            self.rt.apply({"speed": float(text)}, "mqtt")
        elif name == "auto_cycle":
            self.rt.apply({"auto_s": int(float(text))}, "mqtt")
        elif name.startswith("param/"):
            parts = name.split("/")
            if len(parts) != 3:
                raise ValueError("bad param topic")
            self.rt.apply({"params": {parts[1]: {parts[2]: float(text)}}}, "mqtt")
        elif name == "reboot":
            self.rt.request_reboot("Home Assistant reboot button")
        elif name == "install":
            err = self.rt.updates.start_github(self.rt.cfg.UPDATE_CHANNEL)
            if err:
                raise ValueError(err)
            self.dirty["update"] = True
        elif name == "check":
            self.rt.updates.check_soon()
        else:
            raise ValueError("unknown command " + name)
        self.dirty["settings"] = True

    # -- discovery --

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
                    "value_template": "{{ value_json.params.%s.%s }}" % (eff, pname),
                    "min": spec[1],
                    "max": spec[2],
                    "step": _step(spec[1], spec[2]),
                    "mode": "slider",
                    "entity_category": "config",
                }
        yield "sensor", "effect_running", {
            "name": "Running effect",
            "state_topic": self.t_effect,
            "json_attributes_topic": self.t_effect_attr,
            "icon": "mdi:animation-play",
        }
        sensors = (
            ("rssi", "Wi-Fi signal", {"device_class": "signal_strength", "unit_of_measurement": "dBm", "state_class": "measurement"}),
            ("heap", "Free memory", {"device_class": "data_size", "unit_of_measurement": "B", "state_class": "measurement"}),
            ("uptime", "Uptime", {"device_class": "duration", "unit_of_measurement": "s", "state_class": "total_increasing"}),
            ("fps", "Frame rate", {"unit_of_measurement": "fps", "state_class": "measurement", "icon": "mdi:filmstrip"}),
            ("cap_pct", "Power cap output", {"unit_of_measurement": "%", "state_class": "measurement", "icon": "mdi:flash-triangle"}),
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
            "payload_press": '{"action":"play","sequence":"default","loops":0,"delay_ms":5000,"ttl_s":300}',
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
        dev = self._device()
        origin = {"name": "tree firmware", "sw_version": self.rt.version, "support_url": "https://github.com/ldn-house/tree"}
        for comp, obj, cfg in self._entities():
            self.hb = time.ticks_ms()
            if obj is not None:
                cfg["unique_id"] = "xmas_tree_" + obj
            cfg["availability_topic"] = self.t_avail
            cfg["device"] = dev
            cfg["origin"] = origin
            yield self._config_topic(comp, obj), json.dumps(cfg), obj

    def _digest(self):
        h = hashlib.sha256()
        for topic, js, _obj in self._configs():
            h.update(topic)
            h.update(js)
        return h.digest()

    async def publish_discovery(self, force=False):
        """Discovery configs are retained, so they only need publishing when
        they change (a new firmware or effect set), on the first connect,
        or when Home Assistant asks (its birth message). Re-publishing them
        on every reconnect made HA re-initialise every entity each time.
        A pass only counts once every publish in it went out: if the session
        drops part-way, nothing is recorded and the next connect publishes
        it all again (review R3-M3). Returns True when a pass completed."""
        digest = self._digest()
        if not force and digest == self.discovered:
            return False
        self.discovered = None  # until this pass completes
        params = []
        n = 0
        for topic, js, obj in self._configs():
            n += 1
            if obj and obj.startswith("param_"):
                params.append(obj)
            if not await self.mqtt.publish(topic, js, retain=True):
                log("ha: discovery interrupted after %d entities; will retry on reconnect" % (n - 1))
                return False
        # Remove entities for params that no longer exist.
        for obj in self.rt.state.get("ha_params"):
            if obj not in params:
                if not await self.mqtt.publish(self._config_topic("number", obj), b"", retain=True):
                    log("ha: discovery cleanup interrupted; will retry on reconnect")
                    return False
        self.rt.state.set_ha_params(params)
        self.discovered = digest
        self.discoveries += 1
        gc.collect()
        log("ha: discovery published (%d entities, %d bytes free after collect)" % (n, hw.mem_free()))
        return True

    # -- outgoing state --

    def light_payload(self):
        st = self.rt.state
        return {
            "state": "ON" if st.get("power") else "OFF",
            "brightness": st.get("brightness"),
            "effect": st.get("effect"),
            "color_mode": "brightness",
        }

    def settings_payload(self):
        st = self.rt.state
        params = {}
        for eff in self.rt.fx.names():
            spec = self.rt.fx.meta(eff)["params"]
            if spec:
                params[eff] = {k: st.param(eff, k, v[0]) for k, v in spec.items()}
        return {"speed": st.get("speed"), "auto_s": st.get("auto_s"), "params": params}

    def diag_payload(self):
        return self.rt.diag()

    async def _publish(self, what):
        if what == "light":
            await self.mqtt.publish(self.t_state, json.dumps(self.light_payload()), retain=True)
        elif what == "settings":
            await self.mqtt.publish(self.t_settings, json.dumps(self.settings_payload()), retain=True)
        elif what == "update":
            await self.mqtt.publish(self.t_update, json.dumps(self.rt.updates.ha_payload()), retain=True)
        elif what == "calib":
            await self.mqtt.publish(self.t_cal_state, json.dumps(self.rt.calib.state_payload()), retain=True)
        elif what == "diag":
            await self.mqtt.publish(self.t_diag, json.dumps(self.diag_payload()), retain=True)
            self.last_diag = time.ticks_ms()

    async def _effect(self, now):
        """Publish the running effect when it changes: auto advance, a new
        selection, a skip past a failed or quarantined effect, the fallback,
        power off ("off") and calibration ("calibration"). At most one
        publish per EFFECT_MIN_MS; a change inside that window goes out as
        soon as it ends, so the last state always lands."""
        label, mode, nxt = self.rt.engine.running()
        if (label, mode) == self.effect_shown:
            return
        if self.effect_at is not None and time.ticks_diff(now, self.effect_at) < EFFECT_MIN_MS:
            return
        self.effect_shown = (label, mode)
        self.effect_at = now
        attrs = {"mode": mode, "selected": self.rt.state.get("effect"), "next_switch_s": nxt}
        await self.mqtt.publish(self.t_effect_attr, json.dumps(attrs), retain=True)
        await self.mqtt.publish(self.t_effect, label, retain=True)

    async def on_connect(self):
        """Every new session: availability, discovery only if it changed,
        and every state again (retained), so nothing is left stale."""
        self.ready = False
        # We subscribed to our own availability before this runs: a broker
        # that kept its retained messages replays it (retain flag set). If
        # nothing comes, it has lost them (restarted without persistence),
        # discovery configs included, so publish those again.
        self.avail_retained = False
        end = time.ticks_add(time.ticks_ms(), RETAINED_WAIT_MS)
        while not self.avail_retained and time.ticks_diff(end, time.ticks_ms()) > 0:
            self.hb = time.ticks_ms()
            await asyncio.sleep_ms(50)
        await self.mqtt.publish(self.t_avail, "online", retain=True)
        await self.publish_discovery(force=not self.avail_retained)
        self._republish()
        # In case the broker publishes an older session's will after this.
        self.online_at = time.ticks_add(time.ticks_ms(), ONLINE_AGAIN_MS)
        self.ready = True

    def _republish(self):
        self.effect_shown = None
        self.effect_at = None
        for k in self.dirty:
            self.dirty[k] = True

    async def run(self):
        while True:
            now = time.ticks_ms()
            self.hb = now
            await asyncio.sleep_ms(200)
            if not self.mqtt.connected or not self.ready:
                continue
            trace.at(trace.HA)
            calib = self.rt.calib
            if self.rediscover_at is not None and time.ticks_diff(now, self.rediscover_at) >= 0:
                self.rediscover_at = None
                log("ha: Home Assistant restarted (or entities changed), re-publishing discovery")
                await self.publish_discovery(force=True)
                self._republish()
            if self.online_at is not None and time.ticks_diff(now, self.online_at) >= 0:
                self.online_at = None
                await self.mqtt.publish(self.t_avail, "online", retain=True)
            if calib.version != self.last_cal or (calib.playing() and time.ticks_diff(now, calib.published_at) >= 1000):
                self.last_cal = calib.version
                calib.published_at = now
                self.dirty["calib"] = True
            if time.ticks_diff(now, self.last_diag) >= DIAG_EVERY_MS and not calib.playing():
                self.dirty["diag"] = True
            await self._effect(now)
            for what in ("light", "calib", "settings", "update", "diag"):
                if self.dirty[what]:
                    self.dirty[what] = False
                    await self._publish(what)
