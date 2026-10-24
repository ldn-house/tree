"""Temporary takeovers: something else (Home Assistant's F1 automation, say)
shows its own pattern for a while, then the tree goes back to what it was
doing. Commands come over MQTT (<prefix>/takeover/set) and HTTP
(POST /takeover); both take the same JSON object:

    {"id": "f1", "effect": "f1_yellow", "params": {"color": "#ffff00"},
     "color": [255, 255, 0], "brightness": 255, "speed": 1.0,
     "priority": 50, "ttl_s": 180}
    {"id": "f1", "action": "refresh", "ttl_s": 180}   extend a running one's
                                         TTL, if it's still there
    {"id": "f1", "action": "end"}        end one (and lift its dismissal)
    {"action": "end_all"}                end every takeover and dismiss them

- effect: any registered effect (AUTO = False ones too), or "off" for a dark
  tree. params are that effect's params (colours as "#RRGGBB", [r, g, b]
  or 0xRRGGBB); unspecified ones take the effect's defaults, never the
  user's saved params. color is shorthand for params.color. brightness
  (0-255) defaults to the user's; speed (0.1-5) defaults to 1.0.
- ttl_s (1..TTL_MAX_S, default TTL_DEFAULT_S) is a dead man's switch: a
  takeover nobody refreshes ends by itself, so a crashed Home Assistant
  never leaves the tree stuck. Refresh by sending the same command again:
  an identical command only extends the TTL, new params are applied to the
  running effect in place, and only a new effect restarts it (or
  "restart": true, to play an effect's opening again).
- priority (0-100, default 50): the highest active takeover shows; on a tie
  the newest. Each id holds one takeover; at most MAX of them.
- Nothing here is persisted: a reboot ends every takeover, and the tree
  boots into its saved state. The user's state (power, brightness, effect,
  params) is never changed by a takeover, so ending one restores it.

The user's own changes while a takeover shows (app.runtime.apply):
- power off, an effect or a colour ends every takeover and dismisses its id:
  further commands for that id are refused until it is ended explicitly
  ({"action": "end"}), the user turns the tree on again, or DISMISS_S pass.
- brightness applies to the user's state and to the takeover showing.
- anything else (speed, params, floor, auto-cycle) changes the user's state
  only, and shows once the takeover ends.
Calibration outranks takeovers: they keep their TTL running underneath and
show again when it ends, if still alive. A takeover shows even while the
user's state is off (the tree turns on for it, and off again afterwards).
"""

import time

from app import fx as fxmod
from app.log import log

TTL_DEFAULT_S = 120
TTL_MAX_S = 3600
DISMISS_S = 4 * 3600
MAX = 4
PRIORITY_DEFAULT = 50
OFF = "off"


def _int(v, what, lo, hi):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ValueError(what + " must be a number")
    v = int(v)
    if not lo <= v <= hi:
        raise ValueError("%s must be %d..%d" % (what, lo, hi))
    return v


class Takeover:
    def __init__(self, tid):
        self.id = tid
        self.effect = None
        self.params = {}
        self.brightness = None
        self.speed = 1.0
        self.priority = PRIORITY_DEFAULT
        self.ttl_s = TTL_DEFAULT_S
        self.expires = 0
        self.seq = 0  # order of the last command, for ties
        self.gen = 0  # bumped when the effect changes: the engine restarts it

    def left_s(self, now):
        return max(0, time.ticks_diff(self.expires, now) // 1000)

    def payload(self, now):
        return {"id": self.id, "effect": self.effect, "params": self.params, "brightness": self.brightness,
                "speed": self.speed, "priority": self.priority, "ttl_s": self.left_s(now)}


class Takeovers:
    def __init__(self, fx, state):
        self.fx = fx
        self.state = state
        self.active = {}
        self.dismissed = {}  # id -> ticks when the dismissal lapses
        self.seq = 0
        self.version = 0  # bumped on every visible change (HA publishes it)
        self.started = 0
        self.ended = 0
        self.refused = 0
        self.last_end = None  # (id, why)

    def _changed(self):
        self.version += 1

    # -- commands --

    def command(self, cmd, source):
        """Apply one command object. Returns a short result dict; raises
        ValueError for a malformed or refused command."""
        if not isinstance(cmd, dict):
            raise ValueError("expected a JSON object")
        action = cmd.get("action", "set")
        if action == "end_all":
            self.end_all("ended by " + source, dismiss=True)
            return {"ended": "all"}
        tid = cmd.get("id", "default")
        if not isinstance(tid, str) or not fxmod.valid_name(tid):
            raise ValueError("id must be a short lowercase name")
        if action == "end":
            self.dismissed.pop(tid, None)
            return {"ended": self.end(tid, "ended by " + source)}
        now = time.ticks_ms()
        if action == "refresh":
            tk = self.active.get(tid)
            if tk is None:
                return {"id": tid, "refreshed": False}
            ttl = _int(cmd.get("ttl_s", tk.ttl_s), "ttl_s", 1, TTL_MAX_S)
            tk.ttl_s = ttl
            tk.expires = time.ticks_add(now, ttl * 1000)
            self._changed()
            return {"id": tid, "refreshed": True, "ttl_s": ttl}
        if action != "set":
            raise ValueError("unknown action %r" % (action,))
        until = self.dismissed.get(tid)
        if until is not None:
            if time.ticks_diff(until, now) > 0:
                self.refused += 1
                raise ValueError("takeover %s was dismissed by the user" % tid)
            del self.dismissed[tid]
        new = self._parse(tid, cmd)
        tk = self.active.get(tid)
        if tk is None:
            if len(self.active) >= MAX:
                self._evict()
            tk = Takeover(tid)
            tk.effect = new["effect"]
            self.active[tid] = tk
            self.started += 1
            log("takeover: %s started (%s, priority %d, ttl %d s, from %s)" % (
                tid, new["effect"], new["priority"], new["ttl_s"], source))
        elif tk.effect != new["effect"] or cmd.get("restart") is True:
            tk.gen += 1
            log("takeover: %s now %s" % (tid, new["effect"]))
        changed = (tk.effect != new["effect"] or cmd.get("restart") is True or tk.params != new["params"] or tk.brightness != new["brightness"]
                   or tk.speed != new["speed"] or tk.priority != new["priority"] or tk.seq == 0)
        tk.effect = new["effect"]
        tk.params = new["params"]
        tk.brightness = new["brightness"]
        tk.speed = new["speed"]
        tk.priority = new["priority"]
        tk.ttl_s = new["ttl_s"]
        tk.expires = time.ticks_add(now, new["ttl_s"] * 1000)
        if changed:
            # A refresh (identical command) keeps its place among ties.
            self.seq += 1
            tk.seq = self.seq
        self._changed()
        return {"id": tid, "effect": tk.effect, "ttl_s": tk.ttl_s, "showing": self.current() is tk}

    def _parse(self, tid, cmd):
        eff = cmd.get("effect")
        if eff != OFF and (not isinstance(eff, str) or eff not in self.fx.names()):
            raise ValueError("unknown effect %r" % (eff,))
        spec = {} if eff == OFF else self.fx.meta(eff)["params"]
        raw = cmd.get("params") or {}
        if not isinstance(raw, dict):
            raise ValueError("params must be an object")
        if "color" in cmd:
            raw = dict(raw)
            raw["color"] = cmd["color"]
        params = {}
        for k, v in raw.items():
            if k not in spec:
                raise ValueError("unknown param %s.%s" % (eff, k))
            s = spec[k]
            if fxmod.is_rgb(s):
                params[k] = fxmod.parse_rgb(v)
            else:
                if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
                    raise ValueError("param %s must be a number" % k)
                params[k] = s[1] if v < s[1] else s[2] if v > s[2] else v
        bri = cmd.get("brightness")
        if bri is not None:
            bri = _int(bri, "brightness", 0, 255)
        speed = cmd.get("speed", 1.0)
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed != speed:
            raise ValueError("speed must be a number")
        speed = float(min(5.0, max(0.1, speed)))
        return {
            "effect": eff,
            "params": params,
            "brightness": bri,
            "speed": speed,
            "priority": _int(cmd.get("priority", PRIORITY_DEFAULT), "priority", 0, 100),
            "ttl_s": _int(cmd.get("ttl_s", TTL_DEFAULT_S), "ttl_s", 1, TTL_MAX_S),
        }

    def _evict(self):
        worst = None
        for tk in self.active.values():
            if worst is None or (tk.priority, tk.seq) < (worst.priority, worst.seq):
                worst = tk
        if worst is not None:
            self.end(worst.id, "evicted: more than %d takeovers" % MAX)

    def end(self, tid, why):
        tk = self.active.pop(tid, None)
        if tk is None:
            return None
        self.ended += 1
        self.last_end = (tid, why)
        log("takeover: %s ended (%s)" % (tid, why))
        self._changed()
        return tid

    def end_all(self, why, dismiss=False):
        now = time.ticks_ms()
        for tid in list(self.active):
            if dismiss:
                self.dismissed[tid] = time.ticks_add(now, DISMISS_S * 1000)
            self.end(tid, why)

    # -- the user's own changes (app.runtime.apply) --

    def user_change(self, changes, source):
        if changes.get("power") is True and self.dismissed:
            self.dismissed = {}
            self._changed()
        if not self.active:
            return
        if changes.get("power") is False or "effect" in changes:
            self.end_all("the user took over (%s)" % source, dismiss=True)
        elif "brightness" in changes:
            tk = self.current()
            if tk is not None and tk.brightness is not None:
                tk.brightness = None  # show the user's new brightness
                self._changed()

    # -- what shows --

    def expire(self, now):
        # Remove outside the traversal; the normal per-frame check needs no
        # snapshot list (including when no takeovers are running).
        while self.active:
            expired = None
            for tid in self.active:
                if time.ticks_diff(now, self.active[tid].expires) >= 0:
                    expired = tid
                    break
            if expired is None:
                break
            self.end(expired, "ttl expired")

    def current(self):
        best = None
        for tid in self.active:
            tk = self.active[tid]
            if best is None or tk.priority > best.priority or (tk.priority == best.priority and tk.seq > best.seq):
                best = tk
        return best

    def brightness(self, tk):
        return self.state.get("brightness") if tk.brightness is None else tk.brightness

    def status(self):
        now = time.ticks_ms()
        tk = self.current()
        dismissed = [t for t, until in self.dismissed.items() if time.ticks_diff(until, now) > 0]
        dismissed.sort()
        return {
            "active": tk is not None,
            "showing": tk.payload(now) if tk else None,
            "takeovers": sorted([t.payload(now) for t in self.active.values()], key=lambda p: -p["priority"]),
            "dismissed": dismissed,
            "started": self.started,
            "ended": self.ended,
            "refused": self.refused,
            "last_end": {"id": self.last_end[0], "why": self.last_end[1]} if self.last_end else None,
        }
