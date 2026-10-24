"""Firmware entry point: one asyncio event loop, no threads.

bootmgr.launch() imports this from the active slot and calls run(info).
Tasks (each restarted if it raises, and watched by the supervisor):

    engine      effects -> output stage -> PIO/DMA driver
    calib       calibration lock/TTL and static frames (player on demand)
    wifi        Wi-Fi state machine
    mqtt, ha    MQTT session and Home Assistant state/commands
    http        control/OTA server (started regardless of Wi-Fi state)
    updates     periodic update check
    saver       debounced state.json writes
    supervisor  watchdog feeding, health, trial confirmation, reboots
"""

import gc
import time

import asyncio

import bootmgr

from app import calib, cfg, coords, driver, engine, fx, ha, hw, httpd, mqtt, net, outstage, recovery, state, supervisor, trace, updates
from app.log import log

# Collect after every GC_THRESHOLD bytes allocated rather than only when the
# heap runs out, so free memory never sinks more than about this far below
# what is live (the raw low-water mark then means something), and the HA
# discovery burst at each MQTT connect doesn't fill the heap with garbage.
GC_THRESHOLD = 32 * 1024
RESTART_LIMIT = 5
RESTART_WINDOW_MS = 60000


class Runtime:
    def __init__(self, info):
        try:
            gc.threshold(GC_THRESHOLD)
        except (AttributeError, ValueError):
            pass
        info["reset_cause"] = recovery.reset_cause(info)
        crumb = trace.take()
        self.info = info
        self.cfg = cfg
        self.n = cfg.LED_COUNT
        self.slot = info.get("slot") or "dev"
        self.slot_dir = bootmgr.slot_dir(self.slot)
        m = info.get("manifest") or {}
        self.manifest = m
        self.version = m.get("version", "dev")
        self.started = time.ticks_ms()
        # Accumulated by the supervisor: a single ticks_diff from boot is
        # ambiguous after half the ticks period (~6 days on the rp2).
        self.uptime_ms = 0
        self.reboot_at = None
        self.reboot_reason = None
        self.restarts = {}
        self.reset_reason = supervisor.take_reason()
        # A watchdog expiry the supervisor didn't decide on: the event loop
        # was blocked. Say where (app.trace).
        self.hang = crumb if info["reset_cause"] == "watchdog" and not self.reset_reason else None
        if self.hang:
            self.reset_reason = trace.describe(self.hang)
            log("boot: watchdog expired: " + self.reset_reason)
        log("tree %s (slot %s, reset %s%s)" % (self.version, self.slot, info.get("reset_cause"),
                                                ": " + self.reset_reason if self.reset_reason else ""))
        if recovery.OUTDATED:
            log("recovery: API %d lacks %s: %s" % (recovery.API, ", ".join(recovery.MISSING), recovery.MESSAGE))
        if cfg.ERROR:
            log("config: %s; using %s" % (cfg.ERROR, cfg.SOURCE))

        self.state = state.State()
        self.state.load()
        bootmgr.feed()
        self.fx = fx.Effects(self.slot_dir)
        if self.state.get("effect") not in self.fx.names() + ["auto"]:
            log("state: effect %s no longer exists, using auto" % self.state.get("effect"))
            self.state.d["effect"] = "auto"

        self.driver = driver.create(self.n, cfg.LED_PIN, cfg.LED_TIMING_NS)
        budget = outstage.channel_budget(self.n, cfg.POWER_LIMIT_A, cfg.LED_MA_PER_CHANNEL, cfg.POWER_BUDGET_PERCENT)
        if cfg.POWER_BUDGET_PERCENT is not None:
            log("power: POWER_BUDGET_PERCENT is deprecated; set POWER_LIMIT_A (amps at 12 V)")
        log("power: limit %.2f A at 12 V (%d mA per channel) = channel sum %d" % (
            outstage.estimated_a(budget, cfg.LED_MA_PER_CHANNEL), cfg.LED_MA_PER_CHANNEL, budget))
        self.out = outstage.create(self.n, budget, cfg.GAMMA, cfg.COLOR_ORDER, cfg.LED_DITHER, cfg.WHITE_BALANCE,
                                   self.state.get("floor"))
        self.state.listeners.append(self._on_state)
        log("output: %s kernels, dither %s, gamma %s, white balance %s, brightness floor %s%%" % (
            "native" if outstage.NATIVE else "python", cfg.LED_DITHER, cfg.GAMMA, cfg.WHITE_BALANCE,
            self.state.get("floor")))
        # Calibration frames: exact, steady levels for the camera.
        self._cal_out = None
        self.engine = engine.Engine(self.n, self.state, self.fx, self.out, self.driver, cfg.FPS)
        self.engine.colour = outstage.ColourFloor(self.out, self.state.get("colour_accuracy"))
        self.engine.coords = coords.load(self.n, self.slot_dir)
        bootmgr.feed()
        self.engine.boot_quarantine(info.get("reset_cause"))

        # Fallback Wi-Fi: the last good config's credentials, else (API 3)
        # those of the config a /config change replaced.
        fallbacks = bootmgr.fallback_configs() if hasattr(bootmgr, "fallback_configs") else [bootmgr.good_config() or {}]
        good = {}
        for c in fallbacks:
            if c.get("WIFI_SSID") and (c.get("WIFI_SSID"), c.get("WIFI_PASSWORD")) != (cfg.WIFI_SSID, cfg.WIFI_PASSWORD):
                good = c
                break
        self.wifi = net.Wifi(cfg.WIFI_SSID, cfg.WIFI_PASSWORD, cfg.HOSTNAME,
                             fallback=(good.get("WIFI_SSID"), good.get("WIFI_PASSWORD")))
        self.mqtt = None
        self.ha = None
        if cfg.MQTT_BROKER:
            self.mqtt = mqtt.MQTT(
                cfg.MQTT_BROKER,
                cfg.MQTT_PORT,
                "xmas_tree_" + hw.unique_id(),
                cfg.MQTT_USER,
                cfg.MQTT_PASSWORD,
                cfg.MQTT_KEEPALIVE,
            )
        self.calib = calib.Calib(self)
        self.updates = updates.Updates(self)
        if self.mqtt is not None:
            self.ha = ha.HA(self, self.mqtt)
            self.engine.on_colour = lambda: self.ha.mark("diag")
        self.http = httpd.Server(cfg.HTTP_PORT, cfg.HTTP_TOKEN)
        from app import api

        api.register(self, self.http)
        self.calib.routes(self.http)

        self.sup = supervisor.Supervisor(self)
        self.sup.heartbeat("engine", self.engine, 5000)
        self.sup.heartbeat("calib", self.calib, 5000)
        self.sup.heartbeat("wifi", self.wifi, 10000)
        self.sup.heartbeat("updates", self.updates, 30000)
        if self.mqtt is not None:
            self.sup.heartbeat("mqtt", self.mqtt, 20000)
            self.sup.heartbeat("ha", self.ha, 20000)
        self.sup.watch("http", lambda: None if self.http.alive() else "server stopped")
        self.sup.watch("tasks", self._tasks_check)

    @property
    def cal_out(self):
        if self._cal_out is None:
            self._cal_out = outstage.create(self.n, self.out.budget, None, cfg.COLOR_ORDER, False, None, 0)
        return self._cal_out

    # -- shared operations --

    def param_specs(self, effect):
        if effect not in self.fx.names():
            raise ValueError("unknown effect " + str(effect))
        return self.fx.meta(effect)["params"]

    def _on_state(self, changed, source):
        if "floor" in changed:
            self.out.set_floor(changed["floor"])

    def apply(self, changes, source):
        """Apply user changes from MQTT or HTTP. Raises ValueError."""
        if changes.get("power") is False:
            self.calib.unlock("power off from " + source)
        return self.state.update(changes, source, effects=self.fx.names(), param_specs=self.param_specs)

    def request_reboot(self, reason, delay_ms=500):
        if self.reboot_at is None:
            self.reboot_reason = reason
            self.reboot_at = time.ticks_add(time.ticks_ms(), delay_ms)

    def uptime_s(self):
        return self.uptime_ms // 1000

    def boot_status(self):
        s = "trial" if self.info.get("trial") else "ok"
        ev = self.info.get("event")
        return (s + "; " + ev)[:250] if ev else s

    def last_error(self):
        errs = self.engine.errors
        if not errs:
            return "none"
        name = sorted(errs)[-1]
        return ("%s: %s" % (name, errs[name]))[:250]

    def led_amps(self):
        """Estimated strip current, amps at 12 V, from the bytes actually
        sent last (after cap, brightness and gamma; calibration frames too)."""
        return round(outstage.estimated_a(sum(self.engine.last), cfg.LED_MA_PER_CHANNEL), 3)

    def brightness_floor(self):
        """The bottom of the brightness slider and why: the user's floor,
        the running effect's colour floor, and the higher of the two."""
        out = self.out
        cf = self.engine.colour
        return {
            "floor": out.floor,
            "colour": out.colour_floor,
            "effective": out.floor_pct(),
            "colour_accuracy": cf.accuracy,
            "colour_pixels": cf.pixels,
            # Last analysed frame, all its steps; and the longest single
            # call, which is how long one frame was held up.
            "colour_ms": round(cf.us / 1000, 1),
            "colour_call_ms_max": round(self.engine.colour_call_us / 1000, 1),
        }

    def diag(self):
        es = self.engine.status()
        return {
            "rssi": self.wifi.rssi(),
            "heap": self.sup.heap_live if self.sup.heap_live is not None else hw.mem_free(),
            "uptime": self.uptime_s(),
            "fps": es["fps"],
            "capped": es["power_capped"],
            "cap_pct": es["cap_factor"],
            "led_amps": self.led_amps(),
            "floor": self.out.floor_pct(),
            "floor_colour": self.out.colour_floor,
            "reset": (self.info.get("reset_cause") or "?") + (": " + self.reset_reason if self.reset_reason else ""),
            "boot": self.boot_status(),
            "ip": self.wifi.ip,
            "error": self.last_error(),
            "effect": es["running"] or "off",
            "calib": self.calib.state,
            "recovery": "outdated (USB install recommended)" if recovery.OUTDATED else "ok (API %d)" % recovery.API,
        }

    def status(self):
        st = self.state
        m = self.manifest
        return {
            "hostname": cfg.HOSTNAME,
            "ip": self.wifi.ip,
            "uptime_s": self.uptime_s(),
            "free_memory": hw.mem_free(),
            "platform": hw.platform(),
            "version": self.version,
            "commit": m.get("commit"),
            "branch": m.get("ref"),
            "build_time": m.get("built"),
            "slot": self.slot,
            "boot": {
                "status": self.boot_status(),
                "trial": bool(self.info.get("trial")),
                "previous": self.info.get("previous"),
                "boots": self.info.get("boots"),
                "events": self.info.get("events"),
                "reset_cause": self.info.get("reset_cause"),
                "reset_reason": self.reset_reason,
                "hang": self.hang,
            },
            "state": st.snapshot(),
            "effect": self.engine.status(),
            "effects": self.fx.names(),
            "overlays": self.fx.overlays(),
            "coords": coords.count(self.engine.coords),
            "driver": self.driver.mode,
            "led_timing_ns": self.driver.timing_ns,
            "power_budget": self.out.budget,
            "brightness_floor": self.brightness_floor(),
            "power": {
                "limit_a": round(outstage.estimated_a(self.out.budget, cfg.LED_MA_PER_CHANNEL), 3),
                "ma_per_channel": cfg.LED_MA_PER_CHANNEL,
                "estimated_a": self.led_amps(),
                "capped": bool(self.out.capped) and not self.engine.dark,
                "volts": 12,
            },
            "recovery": recovery.status(),
            "reserve": {"bytes": bootmgr.reserve_size(), "complete": bootmgr.reserve_size() == bootmgr.RESERVE_BYTES},
            "floor": bootmgr.read_state()["floor"],
            "wifi": self.wifi.status(),
            "mqtt": self.mqtt.status() if self.mqtt else None,
            "ha": {"discoveries": self.ha.discoveries, "offline_seen": self.ha.offline_seen} if self.ha else None,
            "http": {"requests": self.http.requests, "rejected": self.http.rejected, "auth": bool(cfg.HTTP_TOKEN)},
            "calibration": self.calib.state,
            "update": self.updates.status(),
            "supervisor": {"healthy": self.sup.healthy, "problems": self.sup.problems,
                           "heap_live": self.sup.heap_live, "heap_live_min": self.sup.heap_live_min,
                           "heap_low_uncollected": self.sup.heap_low_uncollected, "heap_sites": hw.heap_sites,
                           "gc_threshold": GC_THRESHOLD, "stall_max_ms": self.sup.stall_max_ms,
                           "stall_in": self.sup.stall_in},
            "config_error": cfg.ERROR,
            "config_source": cfg.SOURCE,
            "config_trial": bool(self.info.get("cfg_trial")),
            "wifi_fallback": self.wifi.using_fallback,
            "restarts": self.restarts,
        }

    # -- tasks --

    def _tasks_check(self):
        for name, (count, since) in self.restarts.items():
            if count > RESTART_LIMIT and time.ticks_diff(time.ticks_ms(), since) < RESTART_WINDOW_MS:
                return "%s keeps crashing" % name
        return None

    async def _guard(self, name, fn):
        while True:
            try:
                await fn()
                log("task %s returned" % name)
            except asyncio.CancelledError:
                return
            except Exception as e:
                log("task %s crashed: %r" % (name, e))
                try:
                    import sys

                    sys.print_exception(e)
                except Exception:
                    pass
            count, since = self.restarts.get(name, (0, time.ticks_ms()))
            if time.ticks_diff(time.ticks_ms(), since) > RESTART_WINDOW_MS:
                count, since = 0, time.ticks_ms()
            self.restarts[name] = (count + 1, since)
            await asyncio.sleep_ms(1000)

    async def _saver(self):
        while True:
            await asyncio.sleep_ms(500)
            # Not during calibration playback: a flash write can stall
            # everything on the Pico, and the frames are timed to the
            # millisecond. The change is saved as soon as playback ends.
            if self.state.due() and not self.calib.playing():
                self.state.save_now()

    async def main(self):
        tasks = [
            ("supervisor", self.sup.run),
            ("engine", self.engine.run),
            ("calib", self.calib.run),
            ("wifi", self.wifi.run),
            ("updates", self.updates.run),
            ("saver", self._saver),
        ]
        if self.mqtt is not None:
            tasks.append(("mqtt", lambda: self.mqtt.run(self.wifi)))
            tasks.append(("ha", self.ha.run))
        for name, fn in tasks:
            asyncio.create_task(self._guard(name, fn))
        trace.start()
        while True:
            try:
                await self.http.start()
                break
            except Exception as e:
                log("http: cannot start server: %r" % (e,))
                await asyncio.sleep_ms(2000)
        gc.collect()
        log("runtime: up, %d bytes free" % hw.mem_free())
        while True:
            await asyncio.sleep_ms(10000)


def run(info):
    rt = Runtime(info)
    try:
        asyncio.run(rt.main())
    except KeyboardInterrupt:  # someone took over at the REPL (mpremote)
        recovery.deliberate()  # API 2's main.py doesn't
        raise
    finally:
        try:
            rt.state.save_now()
            rt.driver.write(bytearray(3 * rt.n))
        except Exception:
            pass
