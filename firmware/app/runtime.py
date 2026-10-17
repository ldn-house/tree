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

from app import calib, cfg, coords, driver, engine, fx, ha, hw, httpd, mqtt, net, outstage, state, supervisor, updates
from app.log import log

RESTART_LIMIT = 5
RESTART_WINDOW_MS = 60000


class Runtime:
    def __init__(self, info):
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
        log("tree %s (slot %s, reset %s%s)" % (self.version, self.slot, info.get("reset_cause"),
                                                ": " + self.reset_reason if self.reset_reason else ""))
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
        budget = outstage.channel_budget(self.n, cfg.POWER_BUDGET_MA, cfg.LED_MA_PER_CHANNEL, cfg.POWER_BUDGET_PERCENT)
        log("power: budget %d mA at 12 V (%d mA per channel) = channel sum %d" % (
            outstage.estimated_ma(budget, cfg.LED_MA_PER_CHANNEL), cfg.LED_MA_PER_CHANNEL, budget))
        self.out = outstage.create(self.n, budget, cfg.GAMMA, cfg.COLOR_ORDER)
        self.cal_out = outstage.create(self.n, budget, None, cfg.COLOR_ORDER)
        self.engine = engine.Engine(self.n, self.state, self.fx, self.out, self.driver, cfg.FPS)
        self.engine.coords = coords.load(self.n, self.slot_dir)
        bootmgr.feed()
        self.engine.boot_quarantine(info.get("reset_cause"))

        self.wifi = net.Wifi(cfg.WIFI_SSID, cfg.WIFI_PASSWORD, cfg.HOSTNAME)
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

    # -- shared operations --

    def param_specs(self, effect):
        if effect not in self.fx.names():
            raise ValueError("unknown effect " + str(effect))
        return self.fx.meta(effect)["params"]

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

    def led_ma(self):
        """Estimated strip current for the last effect frame, mA at 12 V."""
        if self.engine.dark or self.engine.owner is not None:
            return 0
        out = self.out
        shown = out.raw_total * out.factor // 4096 * self.state.get("brightness") // 255
        return outstage.estimated_ma(shown, cfg.LED_MA_PER_CHANNEL)

    def diag(self):
        es = self.engine.status()
        return {
            "rssi": self.wifi.rssi(),
            "heap": hw.mem_free(),
            "uptime": self.uptime_s(),
            "fps": es["fps"],
            "capped": es["power_capped"],
            "cap_pct": es["cap_factor"],
            "led_ma": self.led_ma(),
            "reset": (self.info.get("reset_cause") or "?") + (": " + self.reset_reason if self.reset_reason else ""),
            "boot": self.boot_status(),
            "ip": self.wifi.ip,
            "error": self.last_error(),
            "effect": es["running"] or "off",
            "calib": self.calib.state,
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
            },
            "state": st.snapshot(),
            "effect": self.engine.status(),
            "effects": self.fx.names(),
            "overlays": self.fx.overlays(),
            "coords": coords.count(self.engine.coords),
            "driver": self.driver.mode,
            "led_timing_ns": self.driver.timing_ns,
            "power_budget": self.out.budget,
            "power": {
                "budget_ma": outstage.estimated_ma(self.out.budget, cfg.LED_MA_PER_CHANNEL),
                "ma_per_channel": cfg.LED_MA_PER_CHANNEL,
                "estimated_ma": self.led_ma(),
                "volts": 12,
            },
            "wifi": self.wifi.status(),
            "mqtt": self.mqtt.status() if self.mqtt else None,
            "http": {"requests": self.http.requests, "rejected": self.http.rejected, "auth": bool(cfg.HTTP_TOKEN)},
            "calibration": self.calib.state,
            "update": self.updates.status(),
            "supervisor": {"healthy": self.sup.healthy, "problems": self.sup.problems, "heap_min": self.sup.heap_min,
                           "heap_live_min": self.sup.heap_live_min},
            "config_error": cfg.ERROR,
            "config_source": cfg.SOURCE,
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
            if self.state.due():
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
    finally:
        try:
            rt.state.save_now()
            rt.driver.write(bytearray(3 * rt.n))
        except Exception:
            pass
