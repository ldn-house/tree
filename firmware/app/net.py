"""Wi-Fi as a non-blocking state machine (review H2).

wlan.connect() is issued once and then polled; nothing here waits on the
network. Failures back off from 2 s to 60 s, and every few failures the
interface is power-cycled in separate steps. The hostname is set before the
interface comes up so the rp2 lwIP mDNS responder answers for <hostname>.local
(the hand-rolled responder thread is gone, review C3).
"""

import time

import asyncio

from app.log import log

try:
    import network
except ImportError:  # pragma: no cover
    network = None

CONNECT_TIMEOUT_MS = 20000
BACKOFF_MIN_MS = 2000
BACKOFF_MAX_MS = 60000
RESET_EVERY = 4

# cyw43 link status codes that mean "this attempt is over".
_FAILED = (-1, -2, -3)


class Wifi:
    def __init__(self, ssid, password, hostname):
        self.ssid = ssid
        self.password = password
        self.hostname = hostname
        self.wlan = None
        self.state = "off" if not ssid else "init"
        self.up = False
        self.ip = None
        self.since = time.ticks_ms()
        self.attempts = 0
        self.failures = 0
        self.connects = 0
        self.backoff = BACKOFF_MIN_MS
        self.next_at = 0
        self.hb = time.ticks_ms()
        self.listeners = []

    def rssi(self):
        try:
            return self.wlan.status("rssi") if self.up else None
        except Exception:
            return None

    def _set_up(self, up):
        if up == self.up:
            return
        self.up = up
        if up:
            try:
                self.ip = self.wlan.ifconfig()[0]
            except Exception:
                self.ip = None
            self.connects += 1
            log("wifi: connected, ip %s" % self.ip)
        else:
            log("wifi: down")
            self.ip = None
        for fn in self.listeners:
            try:
                fn(up)
            except Exception as e:
                log("wifi listener failed: %r" % (e,))

    def _start(self):
        if network is None:
            self.state = "off"
            return
        try:
            network.hostname(self.hostname)
        except Exception:
            pass
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)
        try:
            self.wlan.config(hostname=self.hostname)
        except Exception:
            pass
        self.state = "idle"

    def _connect(self, now):
        self.attempts += 1
        try:
            if self.attempts > 1:
                self.wlan.disconnect()
            self.wlan.connect(self.ssid, self.password)
            self.state = "connecting"
            self.since = now
            log("wifi: connecting to %s (attempt %d)" % (self.ssid, self.attempts))
        except Exception as e:
            log("wifi: connect() failed: %r" % (e,))
            self._fail(now)

    def _fail(self, now):
        self.failures += 1
        self.state = "backoff"
        self.next_at = time.ticks_add(now, self.backoff)
        log("wifi: retry in %d s" % (self.backoff // 1000))
        self.backoff = min(self.backoff * 2, BACKOFF_MAX_MS)

    def step(self, now):
        """One state machine step; returns how long to wait before the next."""
        st = self.state
        if st == "off":
            return 1000
        if st == "init":
            self._start()
            return 100
        connected = self.wlan.isconnected()
        if st == "up":
            if not connected:
                self._set_up(False)
                self.state = "idle"
                self.backoff = BACKOFF_MIN_MS
            return 1000
        if connected:
            self.state = "up"
            self.backoff = BACKOFF_MIN_MS
            self._set_up(True)
            return 1000
        if st == "idle":
            self._connect(now)
            return 250
        if st == "connecting":
            status = self.wlan.status()
            if status in _FAILED or time.ticks_diff(now, self.since) > CONNECT_TIMEOUT_MS:
                log("wifi: attempt failed (status %s)" % (status,))
                self._fail(now)
            return 250
        if st == "backoff":
            if time.ticks_diff(now, self.next_at) >= 0:
                if self.failures % RESET_EVERY == 0:
                    log("wifi: power-cycling the interface")
                    self.wlan.active(False)
                    self.state = "reactivate"
                    return 500
                self._connect(now)
            return 250
        if st == "reactivate":
            self.wlan.active(True)
            self.state = "idle"
            return 250
        return 1000

    async def run(self):
        while True:
            now = time.ticks_ms()
            self.hb = now
            wait = self.step(now)
            # Sleep in short slices so the heartbeat stays fresh.
            while wait > 0:
                await asyncio.sleep_ms(min(wait, 500))
                wait -= 500
                self.hb = time.ticks_ms()

    def status(self):
        return {"state": self.state, "ip": self.ip, "rssi": self.rssi(), "connects": self.connects, "failures": self.failures}
