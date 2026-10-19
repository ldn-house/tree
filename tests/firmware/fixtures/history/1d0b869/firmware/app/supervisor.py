"""Watchdog supervisor and health checks.

The hardware watchdog is fed only from here, and only while every watched
task has shown a recent heartbeat (or passes its check). A wedged task, or
a blocked event loop, stops the feeding and the board resets within 8 s.
Before letting that happen the reason is written to /reset_reason.txt so the
next boot can report it.

Health, for the boot counter (bootmgr):
- normal boot: up for HEALTHY_S with every check passing;
- trial boot (just updated): additionally Wi-Fi up, the HTTP server
  listening and MQTT connected (when configured), all at once after
  cfg.TRIAL_HEALTHY_S. If that has not happened by cfg.TRIAL_DEADLINE_S the
  supervisor reboots, and bootmgr rolls back to the previous firmware.
"""

import gc
import time

import asyncio

import bootmgr

from app import hw
from app.log import log

REASON_FILE = "reset_reason.txt"
HEALTHY_S = 30
PERIOD_MS = 500
STARTUP_GRACE_MS = 5000
PERSIST_MS = 2000  # a problem must last this long before feeding stops
GC_EVERY_MS = 10000


def take_reason():
    try:
        with open(REASON_FILE) as f:
            why = f.read().strip()
        import os

        os.remove(REASON_FILE)
        return why or None
    except OSError:
        return None


def write_reason(why):
    try:
        with open(REASON_FILE, "w") as f:
            f.write(why)
    except OSError:
        pass


class Supervisor:
    def __init__(self, rt):
        self.rt = rt
        self.checks = []
        self.failing = None
        self.healthy = False
        self.confirmed = False
        self.feeds = 0
        self.problems = []
        self.gave_up = False
        self.heap_min = None  # lowest gc.mem_free() seen, sampled each loop
        # Lowest free heap right after a collection: the real headroom
        # (heap_min mostly measures garbage not collected yet).
        self.heap_live_min = None
        self._gc_at = 0

    def heartbeat(self, name, obj, max_ms):
        def check():
            age = time.ticks_diff(time.ticks_ms(), obj.hb)
            return None if age < max_ms else "no heartbeat for %d ms" % age

        self.checks.append((name, check))

    def watch(self, name, fn):
        self.checks.append((name, fn))

    def _problems(self):
        out = []
        for name, fn in self.checks:
            try:
                why = fn()
            except Exception as e:
                why = "check failed: %r" % (e,)
            if why:
                out.append("%s: %s" % (name, why))
        return out

    def _trial_ready(self):
        rt = self.rt
        if rt.cfg.WIFI_SSID and not rt.wifi.up:
            return False
        if not rt.http.alive():
            return False
        if rt.mqtt is not None and not rt.mqtt.connected:
            return False
        return True

    def _health(self, uptime_s):
        rt = self.rt
        if rt.info.get("trial") and not self.confirmed:
            if uptime_s >= rt.cfg.TRIAL_HEALTHY_S and self._trial_ready():
                self.confirmed = True
                self.healthy = True
                if bootmgr.mark_healthy(rt.slot):
                    log("supervisor: update %s confirmed healthy" % rt.version)
                    rt.info["event"] = "confirmed %s" % rt.version
                rt.info["trial"] = False
                if rt.ha:
                    rt.ha.mark("diag")
            elif uptime_s >= rt.cfg.TRIAL_DEADLINE_S:
                rt.request_reboot("update not healthy after %d s; rolling back" % uptime_s)
            return
        if not self.healthy and uptime_s >= HEALTHY_S:
            self.healthy = True
            bootmgr.mark_healthy(rt.slot)
            log("supervisor: healthy")

    async def run(self):
        wdt = bootmgr.watchdog()
        last = time.ticks_ms()
        while True:
            rt = self.rt
            now = time.ticks_ms()
            rt.uptime_ms += time.ticks_diff(now, last)
            last = now
            free = hw.mem_free()
            if self.heap_min is None or free < self.heap_min:
                self.heap_min = free
            if rt.uptime_ms - self._gc_at >= GC_EVERY_MS and not rt.calib.playing():
                self._gc_at = rt.uptime_ms
                gc.collect()
                free = hw.mem_free()
                if self.heap_live_min is None or free < self.heap_live_min:
                    self.heap_live_min = free
            problems = self._problems() if rt.uptime_ms > STARTUP_GRACE_MS else []
            self.problems = problems
            if problems and self.failing is None:
                self.failing = now
            elif not problems:
                self.failing = None
            failed = self.failing is not None and time.ticks_diff(now, self.failing) >= PERSIST_MS
            if failed:
                if not self.gave_up:
                    self.gave_up = True
                    why = "; ".join(problems)
                    log("supervisor: %s; letting the watchdog reset the board" % why)
                    write_reason("watchdog: " + why)
                    rt.state.save_now()
            else:
                if wdt is not None:
                    wdt.feed()
                self.feeds += 1
                if not problems:
                    self._health(rt.uptime_ms // 1000)
            if rt.reboot_at is not None and time.ticks_diff(now, rt.reboot_at) >= 0:
                await self._reboot()
            await asyncio.sleep_ms(PERIOD_MS)

    async def _reboot(self):
        rt = self.rt
        log("supervisor: rebooting: " + rt.reboot_reason)
        write_reason("requested: " + rt.reboot_reason)
        rt.state.save_now()
        rt.engine.clear_probe()
        hw.scratch_set(None)
        if rt.mqtt is not None and rt.mqtt.connected and rt.ha is not None:
            try:
                await asyncio.wait_for(rt.mqtt.publish(rt.ha.t_avail, "offline", retain=True), 1)
            except Exception:
                pass
        try:
            await asyncio.wait_for(rt.engine.present(bytearray(3 * rt.n), rt.out, 0), 1)
        except Exception:
            pass
        await asyncio.sleep_ms(100)
        hw.reset()
