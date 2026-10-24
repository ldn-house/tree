"""User-facing state, persisted to flash and authoritative at boot.

Power, brightness, the brightness floor, colour accuracy, effect, speed,
auto-cycle duration and effect params are saved (debounced, atomically) to /state.json and restored at boot, whatever
the network is doing. MQTT and HTTP only ever send changes through update();
retained MQTT messages never override what is restored here (review M4).
"""

import json
import time

import bootmgr

from app import outstage, trace
from app.log import log

FILE = "state.json"
SAVE_DELAY_MS = 2000

SPEED_MIN = 0.1
SPEED_MAX = 5.0
AUTO_MAX_S = 3600
# Brightness floor, percent (effects/_output.py): non-zero brightness maps
# onto floor..100%. A state setting rather than a config key, so it works
# (and can be tuned from Home Assistant) on any recovery layer.
FLOOR_MAX = 25


def defaults():
    return {
        "power": True,
        "brightness": 128,
        "floor": outstage.FLOOR_PCT,
        # u'v' drift allowed at the bottom of the slider (the colour floor,
        # effects/_output.py); 0 turns it off.
        "colour_accuracy": outstage.COLOUR_ACCURACY,
        "effect": "auto",
        "speed": 1.0,
        "transition_style": "fade",
        "transition_s": 0.5,
        "auto_s": 0,  # 0: use each effect's DURATION_MS
        "params": {},
        "ha_params": [],  # param entities last published to HA discovery
    }


def _num(v):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError("not a number")
    if v != v or v in (float("inf"), float("-inf")):
        raise ValueError("not finite")
    return v


def clamp(v, lo, hi):
    return lo if v < lo else hi if v > hi else v


class State:
    def __init__(self):
        self.d = defaults()
        self.listeners = []
        self._dirty_at = None
        self.saves = 0

    def load(self):
        try:
            with open(FILE) as f:
                data = json.load(f)
        except OSError:
            log("state: no saved state, using defaults")
            return
        except Exception as e:
            log("state: unreadable state.json (%r), using defaults" % (e,))
            return
        if not isinstance(data, dict):
            return
        try:
            self.update(data, source="flash", notify=False, save=False)
        except Exception as e:
            log("state: bad saved state (%r)" % (e,))
        if isinstance(data.get("ha_params"), list):
            self.d["ha_params"] = [p for p in data["ha_params"] if isinstance(p, str)]

    def get(self, key):
        return self.d[key]

    def param(self, effect, name, default):
        return self.d["params"].get(effect, {}).get(name, default)

    def update(self, changes, source="?", notify=True, save=True, effects=None, param_specs=None):
        """Validate and apply changes; returns the dict of what changed.
        Raises ValueError for a malformed value. effects: allowed effect
        names; param_specs: callable(effect) -> PARAMS dict."""
        if not isinstance(changes, dict):
            raise ValueError("expected an object")
        applied = {}
        if "power" in changes:
            v = changes["power"]
            if not isinstance(v, bool):
                raise ValueError("power must be true/false")
            applied["power"] = v
        if "brightness" in changes:
            v = _num(changes["brightness"])
            applied["brightness"] = int(clamp(v, 0, 255))
        if "floor" in changes:
            v = float(clamp(_num(changes["floor"]), 0, FLOOR_MAX))
            applied["floor"] = int(v) if v == int(v) else v
        if "colour_accuracy" in changes:
            applied["colour_accuracy"] = round(outstage.colour_accuracy(float(_num(changes["colour_accuracy"]))), 4)
        if "effect" in changes:
            v = changes["effect"]
            if not isinstance(v, str) or not v or len(v) > 32:
                raise ValueError("bad effect name")
            if effects is not None and v != "auto" and v not in effects:
                raise ValueError("unknown effect: " + v)
            applied["effect"] = v
        if "transition_style" in changes:
            from app.transitions import STYLES
            if changes["transition_style"] not in STYLES:
                raise ValueError("bad transition style")
            applied["transition_style"] = changes["transition_style"]
        if "transition_s" in changes:
            applied["transition_s"] = float(clamp(_num(changes["transition_s"]), 0, 60))
        transition_ms = None
        if "transition" in changes:
            transition_ms = int(clamp(_num(changes["transition"]), 0, 60) * 1000)
        if "speed" in changes:
            applied["speed"] = float(clamp(_num(changes["speed"]), SPEED_MIN, SPEED_MAX))
        if "auto_s" in changes:
            applied["auto_s"] = int(clamp(_num(changes["auto_s"]), 0, AUTO_MAX_S))
        if "params" in changes:
            p = changes["params"]
            if not isinstance(p, dict):
                raise ValueError("params must be an object")
            merged = {}
            for eff, vals in self.d["params"].items():
                merged[eff] = dict(vals)
            for eff, vals in p.items():
                if not isinstance(vals, dict):
                    raise ValueError("params." + str(eff) + " must be an object")
                spec = param_specs(eff) if param_specs else None
                cur = merged.setdefault(eff, {})
                for name, v in vals.items():
                    v = _num(v)
                    if spec is not None:
                        if name not in spec:
                            raise ValueError("unknown param %s.%s" % (eff, name))
                        lo, hi = spec[name][1], spec[name][2]
                        v = clamp(v, lo, hi)
                    cur[name] = v
            applied["params"] = merged
        changed = {}
        for k, v in applied.items():
            if self.d.get(k) != v:
                self.d[k] = v
                changed[k] = v
        if changed and save:
            self._dirty_at = time.ticks_ms()
        # Commands can replace an in-flight display ramp without changing
        # persisted targets. Keep command notifications separate from changes.
        event = dict(changed)
        if transition_ms is not None:
            for k in ("power", "brightness"):
                if k in applied:
                    event[k] = applied[k]
            if event:
                event["transition_ms"] = transition_ms
        if event and notify:
            for fn in self.listeners:
                try:
                    fn(event, source)
                except Exception as e:
                    log("state listener failed: %r" % (e,))
        return changed

    def set_ha_params(self, ids):
        if ids != self.d["ha_params"]:
            self.d["ha_params"] = ids
            self._dirty_at = time.ticks_ms()

    def snapshot(self):
        return {k: v for k, v in self.d.items() if k != "ha_params"}

    def save_now(self):
        if self._dirty_at is None:
            return
        trace.at(trace.STATE_SAVE)
        try:
            bootmgr.write_atomic(FILE, json.dumps(self.d).encode())
            self.saves += 1
            self._dirty_at = None
        except Exception as e:
            log("state: save failed: %r" % (e,))

    def due(self):
        return self._dirty_at is not None and time.ticks_diff(time.ticks_ms(), self._dirty_at) >= SAVE_DELAY_MS
