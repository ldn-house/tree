"""Render loop: schedules effects and pushes frames through the output stage.

The runtime owns the clock, frame pacing, brightness and power cap; effects
only fill a raw RGB frame (docs/effect-api.md). Nothing an effect does can
take the runtime down:

- An exception in import, __init__ or render() marks that effect failed and
  the scheduler moves on. After MAX_FAILS it is skipped until reboot or
  until it is reselected.
- The colour floor (effects/_output.py, ColourFloor) is found from the
  first frame after an effect starts, its params change or the coordinates
  change, then refined from frames 1, 3 and 6 s later. Each analysis is
  spread over the following frames, at most COLOUR_US per frame (beyond
  one colour, which can't be split), and the previous floor stays until the
  first is done. Nothing runs per frame otherwise.
- An effect that hangs the event loop starves the watchdog. Two records say
  which effect it was: watchdog scratch 0 (set whenever an effect starts,
  survives the reset) and /fx_probe.txt (written before an effect's first
  frames each boot). Either way it is quarantined for the next boot instead
  of causing a reboot loop, but only after a genuine watchdog expiry
  (bootmgr.reset_cause() == "watchdog"); every deliberate reset clears both
  records first (app.recovery.deliberate()).
"""

import gc
import random
import sys
import time

import asyncio

from app import fallback, hw, trace
from app.transitions import Blend, Ramp
from app.log import log

PROBE = "fx_probe.txt"
PROVE_FRAMES = 30
MAX_FAILS = 3
SLOW_US = 500000
FALLBACK = "fallback"
BLACK_REFRESH_MS = 2000
T_RESTART_MS = 1 << 29  # ~6 days
# Colour floor: analysis time per frame, at most (a frame is 25 ms at 40 fps;
# one colour takes about 10 ms on the Pico and isn't split).
COLOUR_US = 4000


class Ctx:
    def __init__(self, n, coords, params, rng):
        self.n = n
        self.coords = coords
        self.params = params
        self.random = rng


def _ctx(n, coords, params):
    try:
        import effects

        return effects.Ctx(n, coords, params, random)
    except Exception:
        return Ctx(n, coords, params, random)


class Engine:
    def __init__(self, n, state, fx, out, driver, fps):
        self.n = n
        self.state = state
        self.fx = fx
        self.out = out
        self.driver = driver
        self.period_ms = max(5, 1000 // max(1, fps))
        self.raw = bytearray(3 * n)
        self.bufs = (bytearray(3 * n), bytearray(3 * n))
        self.bi = 0
        self.last = self.bufs[1]
        self.blend = Blend(n)
        self.displayed = False
        self.ramp = Ramp(state.get("brightness") if state.get("power") else 0)
        self.coords = None
        self.name = None
        self.mod = None
        self.inst = None
        self.ctx = None
        self.t_ms = 0  # effect clock (scaled by speed), always a small int
        self.t_frac = 0  # sub-millisecond remainder, microseconds
        self.t_last = time.ticks_ms()
        self.frames_ok = 0
        self.slow = 0
        self.probing = False
        self.proven = set()
        self.failed = {}
        self.errors = {}
        self.quarantined = set()
        self.auto_name = None
        self.auto_until = 0
        self.owner = None  # calibration session while it holds the strip
        self.takeovers = None  # app.takeover.Takeovers, set by the runtime
        self.tk = None  # the takeover the running effect belongs to
        self.src = None  # (id, gen) of that takeover; None: the user's state
        self.tk_params = None
        self.reload = False
        self.hb = time.ticks_ms()
        self.fps = 0
        self._fps_n = 0
        self._fps_t = time.ticks_ms()
        self.render_us = 0
        self.cap_frames = 0
        self.dark = False
        self._black_at = None
        self.colour = None  # ColourFloor for out, set by the runtime
        self.colour_t0 = time.ticks_ms()
        self.colour_call_us = 0  # longest single colour floor call (one frame's stall)
        self.on_colour = None  # called when the colour floor changes
        state.listeners.append(self._on_state)

    # -- boot-time quarantine --

    def boot_quarantine(self, reset_cause):
        """Only a watchdog reset can mean an effect hung; after a power cut
        the probe file just records what happened to be starting."""
        names = self.fx.names()
        # Only a genuine expiry ("watchdog": REASON TIMER) can mean a hang.
        # Deliberate resets ("reboot": REASON FORCE) also read WDT_RESET on
        # the rp2, and their markers are cleared beforehand anyway.
        # "wdt" means the recovery layer and the app both couldn't tell:
        # markers are cleared before every deliberate reset, so trust them.
        if reset_cause in ("watchdog", "wdt"):
            try:
                with open(PROBE) as f:
                    name = f.read().strip()
                if name:
                    self._quarantine(name, "it was starting when the watchdog reset the board")
            except OSError:
                pass
            idx = hw.scratch_get()
            if idx is not None and idx < len(names):
                self._quarantine(names[idx], "it was running when the watchdog fired")
        self.clear_probe()
        hw.scratch_set(None)

    def _quarantine(self, name, why):
        if name in self.quarantined:
            return
        self.quarantined.add(name)
        self.errors[name] = "quarantined: " + why
        log("engine: quarantined %s for this boot: %s" % (name, why))

    def clear_probe(self):
        try:
            import os

            os.remove(PROBE)
        except OSError:
            pass
        self.probing = False

    # -- selection --

    def set_coords(self, coords):
        self.coords = coords
        self.blend.coords(coords)
        self.reload = True

    def reload_effect(self, name=None):
        if name is None or name == self.name:
            self.reload = True
        if name is not None:
            self.failed.pop(name, None)
            self.errors.pop(name, None)
            self.quarantined.discard(name)

    def _showing_tk(self, now):
        """The takeover that shows, if any: expired ones end first, and one
        whose effect can't run (failed, quarantined, no coordinates) leaves
        the user's state showing."""
        if self.takeovers is None:
            return None
        self.takeovers.expire(now)
        tk = self.takeovers.current()
        if tk is not None and tk.effect != "off" and not self.eligible(tk.effect):
            return None
        return tk

    def _goal(self, tk):
        """The brightness to show: a takeover's (0 for a dark one) while one
        shows, else the user's (0 when off)."""
        if tk is not None:
            return 0 if tk.effect == "off" else self.takeovers.brightness(tk)
        return self.state.get("brightness") if self.state.get("power") else 0

    def _ramp_to(self, target, now, ms):
        if not target and self.blend.ms:
            # Wire blends cannot be dimmed without changing low-channel hue
            # and speckle. Going dark during a blend cuts on the next frame.
            self.blend.ms = 0
            ms = 0
        self.ramp.set(target, now, ms)
        if target and self.blend.ms:
            # Finish the blend at its displayed brightness, then start the
            # new ramp through Output, including instant commands.
            self.ramp.value = self.ramp.start
            self.ramp.pending = True

    def _on_state(self, changed, source):
        if "power" in changed or "brightness" in changed:
            now = time.ticks_ms()
            ms = changed.get("transition_ms")
            if ms is None and "power" in changed:
                ms = 1000
            self._ramp_to(self._goal(self._showing_tk(now)), now, ms)
        if "effect" in changed:
            eff = changed["effect"]
            # Reselecting an effect gives it another chance.
            self.failed.pop(eff, None)
            self.quarantined.discard(eff)
            if eff == "auto":
                self.auto_until = 0
        if "params" in changed and self.ctx is not None and self.name and self.src is None:
            vals = changed["params"].get(self.name)
            if vals:
                moved = False
                for k, v in vals.items():
                    if k in self.ctx.params:
                        if self.ctx.params[k] != v:
                            moved = True
                        self.ctx.params[k] = v
                if moved:
                    self.colour_reset()
        if "colour_accuracy" in changed and self.colour is not None:
            self.colour.set_accuracy(changed["colour_accuracy"])
            self.colour_reset()

    def colour_reset(self):
        """Find the colour floor again, from the next frame on."""
        if self.colour is not None:
            self.colour.reset()
            self.colour_t0 = time.ticks_ms()

    def _colour(self, now):
        cf = self.colour
        busy = cf.busy
        if not busy and not cf.due(time.ticks_diff(now, self.colour_t0)):
            return
        trace.at(trace.COLOUR)
        before = cf.pct
        t0 = time.ticks_us()
        if not busy:
            cf.start(self.raw)
        done = cf.step(us=COLOUR_US)
        us = time.ticks_diff(time.ticks_us(), t0)
        if us > self.colour_call_us:
            self.colour_call_us = us
        if done and cf.pct != before and self.on_colour is not None:
            self.on_colour()

    def eligible(self, name):
        if name == FALLBACK:
            return True
        if name not in self.fx.names() or name in self.quarantined:
            return False
        if self.failed.get(name, 0) >= MAX_FAILS:
            return False
        if self.coords is None and self.fx.meta(name)["needs_coords"]:
            return False
        return True

    def running(self):
        """(label, mode, next_switch_s) for what the strip is showing now:
        label is the effect, "off" or "calibration"; next_switch_s is the
        time left before auto mode moves on (None when locked)."""
        if self.owner is not None:
            return "calibration", "calibration", None
        tk = self.showing()
        if tk is not None:
            if tk.effect == "off":
                return "off", "takeover", None
            if self.src is not None:
                return self.name, "takeover", None
        if not self.state.get("power"):
            return "off", "off", None
        if self.name is None:
            return "starting", "auto" if self.state.get("effect") == "auto" else "locked", None
        mode = self.mode()
        nxt = None
        if mode != "locked":
            nxt = max(0, time.ticks_diff(self.auto_until, time.ticks_ms()) // 1000)
        return self.name, mode, nxt

    def showing(self):
        """The takeover that owns the strip now (calibration aside)."""
        return self.takeovers.current() if self.takeovers is not None else None

    def mode(self):
        tk = self.showing()
        if self.src is not None or (tk is not None and tk.effect == "off"):
            return "takeover"
        eff = self.state.get("effect")
        if eff != "auto" and self.name == eff:
            return "locked"
        if self.name == FALLBACK:
            return "fallback"
        return "auto"

    def _duration(self, name):
        s = self.state.get("auto_s")
        if s:
            return s * 1000
        if name == FALLBACK:
            return fallback.DURATION_MS
        return max(1000, self.fx.meta(name)["duration_ms"])

    def _target(self, now):
        eff = self.state.get("effect")
        if eff != "auto" and self.eligible(eff):
            return eff
        if self.auto_name is not None and self.eligible(self.auto_name) and time.ticks_diff(self.auto_until, now) > 0:
            return self.auto_name
        names = self.fx.names()
        start = names.index(self.auto_name) + 1 if self.auto_name in names else 0
        pick = FALLBACK
        for i in range(len(names)):
            cand = names[(start + i) % len(names)]
            # AUTO = False effects (docs/effect-api.md) are only shown when
            # selected or by a takeover.
            if self.eligible(cand) and self.fx.meta(cand).get("auto", True):
                pick = cand
                break
        self.auto_name = pick
        self.auto_until = time.ticks_add(now, self._duration(pick))
        return pick

    # -- effect lifecycle --

    def _unload(self):
        name = self.name
        self.tk = None
        self.src = None
        self.tk_params = None
        self.inst = None
        self.ctx = None
        self.mod = None
        self.name = None
        if name and name != FALLBACK:
            self.fx.unload(name)
        gc.collect()

    def _params(self, name, meta, tk=None):
        params = {}
        for k, spec in meta["params"].items():
            if tk is not None:
                # A takeover's own values over the effect's defaults; the
                # user's saved params are theirs, not the takeover's.
                params[k] = tk.params.get(k, spec[0])
            else:
                params[k] = self.state.param(name, k, spec[0])
        return params

    def transition(self, style=None, duration=None):
        """Freeze the last wire frame; explicit none/0 supports instant owners."""
        if style is None:
            style = self.state.get("transition_style")
        if duration is None:
            duration = self.state.get("transition_s")
        # Boot assigns coords directly; later uploads use set_coords().
        if self.blend.points is not self.coords:
            self.blend.coords(self.coords)
        self.blend.begin(self.last, style, int(duration * 1000), time.ticks_ms())

    def _start(self, name, tk=None, transition_style=None, transition_s=None):
        if self.displayed and self.owner is None and name != self.name:
            self.transition(transition_style, transition_s)
        self._unload()
        trace.at(trace.EFFECT_START)
        try:
            if name == FALLBACK:
                mod = fallback
                meta = {"params": {}}
            else:
                names = self.fx.names()
                hw.scratch_set(names.index(name))
                if name not in self.proven:
                    with open(PROBE, "w") as f:
                        f.write(name)
                    self.probing = True
                mod = self.fx.load(name)
                meta = self.fx.meta(name)
            ctx = _ctx(self.n, self.coords, self._params(name, meta, tk))
            inst = mod.Effect(ctx)
        except Exception as e:
            self._fail(name, e)
            return False
        self.tk = tk
        self.src = (tk.id, tk.gen) if tk is not None else None
        self.tk_params = tk.params if tk is not None else None
        self.name = name
        self.mod = mod
        self.ctx = ctx
        self.inst = inst
        self.t_ms = 0
        self.t_frac = 0
        self.t_last = time.ticks_ms()
        self.frames_ok = 0
        self.slow = 0
        self.colour_reset()
        # Drop the constructor's garbage (per-LED tables are built from
        # floats) now, and log what is really free with this effect loaded.
        gc.collect()
        log("engine: running %s (%d bytes free after collect)" % (name, hw.heap_mark("effect start")))
        return True

    def _fail(self, name, e):
        self.failed[name] = self.failed.get(name, 0) + 1
        self.errors[name] = repr(e)
        log("engine: effect %s failed (%d/%d): %r" % (name, self.failed[name], MAX_FAILS, e))
        try:
            sys.print_exception(e)
        except Exception:
            pass
        if self.probing:
            self.clear_probe()
        if self.name == name:
            self._unload()
        if self.auto_name == name:
            self.auto_until = 0

    # -- output --

    def process(self, raw, out, buf, brightness, now):
        """Synchronous per-frame output path; allocation-checked on MicroPython."""
        out.process(raw, buf, brightness)
        if self.owner is None:
            self.blend.apply(buf, now, out.budget)

    async def present(self, raw, out, brightness):
        """Run raw through the output stage and send it (double-buffered)."""
        buf = self.bufs[self.bi]
        trace.at(trace.OUTPUT)
        self.process(raw, out, buf, brightness, time.ticks_ms())
        while self.driver.busy():
            await asyncio.sleep_ms(1)
        self.driver.write(buf)
        self.last = buf
        self.displayed = self.owner is None
        self.bi ^= 1
        if out.capped:
            self.cap_frames += 1

    async def _black(self, now):
        self.blend.ms = 0
        if self._black_at is None or time.ticks_diff(now, self._black_at) >= BLACK_REFRESH_MS:
            for i in range(len(self.raw)):
                self.raw[i] = 0
            await self.present(self.raw, self.out, 0)
            self.displayed = False
            self._black_at = now

    def _tick_fps(self, now):
        self._fps_n += 1
        dt = time.ticks_diff(now, self._fps_t)
        if dt >= 1000:
            self.fps = self._fps_n * 1000 // dt
            self._fps_n = 0
            self._fps_t = now

    async def run(self):
        while True:
            now = time.ticks_ms()
            self.hb = now
            if self.owner is not None:
                self.blend.ms = 0
                self.displayed = False
                self.ramp.set(self._goal(self._showing_tk(now)), now, 0)
                if self.inst is not None:
                    self._unload()
                self.fps = 0
                await asyncio.sleep_ms(50)
                continue
            tk = self._showing_tk(now)
            goal = self._goal(tk)
            if goal != self.ramp.target:
                # A takeover started, ended or changed brightness (the
                # user's own power and brightness arrive through _on_state).
                self._ramp_to(goal, now, None if goal and self.ramp.target else 1000)
            bri = self.ramp.value if self.blend.ms and self.ramp.pending else self.ramp.step(now)
            dark_tk = tk is not None and tk.effect == "off"
            if goal == 0 and bri == 0:
                if self.inst is not None:
                    self._unload()
                self.dark = True
                self.fps = 0
                await self._black(now)
                await asyncio.sleep_ms(50 if dark_tk else 100)
                continue
            self.dark = False
            self._black_at = None
            if tk is not None and not dark_tk:
                want = tk.effect
                src = (tk.id, tk.gen)
                show = tk
            elif goal or self.name is None:
                want = self._target(now)
                src = None
                show = None
            else:
                # Fading out (power off or a dark takeover): keep what shows.
                want = self.name
                src = self.src
                show = self.tk
            if self.reload or want != self.name or src != self.src or self.inst is None:
                self.reload = False
                if not self._start(want, show):
                    await asyncio.sleep_ms(10)
                    continue
            elif show is not None and show.params is not self.tk_params:
                # Same effect, new values: in place, like the user's params.
                self.tk_params = show.params
                meta = self.fx.meta(want)["params"]
                moved = False
                for k, spec in meta.items():
                    v = show.params.get(k, spec[0])
                    if self.ctx.params.get(k) != v:
                        moved = True
                    self.ctx.params[k] = v
                if moved:
                    self.colour_reset()
            await self._frame(now)
            elapsed = time.ticks_diff(time.ticks_ms(), now)
            await asyncio.sleep_ms(max(1, self.period_ms - elapsed))

    async def _frame(self, now):
        dt = time.ticks_diff(now, self.t_last)
        self.t_last = now
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        # Accumulate scaled time so t_ms stays monotonic when speed changes.
        speed = self.tk.speed if self.tk is not None else self.state.get("speed")
        frac = self.t_frac + int(dt * 1000 * speed)
        t = self.t_ms + frac // 1000
        if t >= T_RESTART_MS:
            # Restart instead of rendering: render() never sees t_ms at or
            # past the contract's limit (docs/effect-api.md).
            self.reload = True
            return
        self.t_ms = t
        self.t_frac = frac % 1000
        t0 = time.ticks_us()
        trace.at(trace.RENDER)
        try:
            self.inst.render(self.t_ms, self.raw)
        except Exception as e:
            self._fail(self.name, e)
            return
        self.render_us = time.ticks_diff(time.ticks_us(), t0)
        if self.render_us > SLOW_US:
            self.slow += 1
            if self.slow >= 3:
                self._fail(self.name, RuntimeError("render took %d ms" % (self.render_us // 1000)))
                return
        else:
            self.slow = 0
        self.frames_ok += 1
        if self.colour is not None:
            self._colour(now)
        if self.frames_ok == PROVE_FRAMES and self.name not in self.proven:
            self.proven.add(self.name)
            if self.probing:
                self.clear_probe()
        await self.present(self.raw, self.out, self.ramp.value)
        self._tick_fps(now)

    def status(self):
        return {
            "running": self.name,
            "mode": self.mode() if self.owner is None else "calibration",
            "takeover": self.tk.id if self.tk is not None else None,
            "fps": self.fps,
            "render_ms": self.render_us // 1000,
            "power_capped": bool(self.out.capped) and not self.dark,
            "cap_factor": self.out.factor * 100 // 4096,
            "capped_frames": self.cap_frames,
            "failed": dict(self.errors),
            "quarantined": sorted(self.quarantined),
        }
