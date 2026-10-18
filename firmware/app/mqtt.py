"""Small asyncio MQTT 3.1.1 client (QoS 0 publish, QoS 0/1 receive).

Written for this firmware rather than vendoring mqtt_as because mqtt_as also
manages Wi-Fi, which here belongs to app.net. What matters for reliability:

- connect is non-blocking with hard timeouts on TCP connect and CONNACK
  (review H2);
- every read error, EOF or protocol error ends the session, and so does
  silence for 1.5 x keepalive, so a dead broker is always noticed (H1);
- PINGREQ goes out every keepalive / 2 whatever else is being sent: the
  broker only ever *sends* when asked (or for our subscriptions), so pinging
  only when idle starved last_rx on a tree that publishes all the time and
  the session was dropped every 45 s;
- reconnects back off from 1 s to 60 s, never blocking the event loop;
- the last will marks the device offline when the session dies;
- one task reads; writes go through a lock so packets never interleave.
"""

import time

import asyncio

from app import dns, hw, trace
from app.log import log
from app.streams import close_now

CONNECT_TIMEOUT = 5
WRITE_TIMEOUT = 5
MAX_PACKET = 4096


def _str(s):
    b = s.encode() if isinstance(s, str) else s
    return len(b).to_bytes(2, "big") + b


def _packet(kind, body):
    n = len(body)
    out = bytearray([kind])
    while True:
        b = n & 0x7F
        n >>= 7
        out.append(b | 0x80 if n else b)
        if not n:
            break
    out.extend(body)
    return out


async def _timeout(coro, s):
    return await asyncio.wait_for(coro, s)


class MQTT:
    def __init__(self, host, port, client_id, user=None, password=None, keepalive=30, will=None):
        self.host = host
        self.port = port
        self.client_id = client_id
        self.user = user
        self.password = password
        self.keepalive = keepalive
        self.will = will  # (topic, payload, retain)
        self.subs = []
        self.on_connect = None  # async callable
        self.on_message = None  # callable(topic, payload, retained)
        self.connected = False
        self.r = None
        self.w = None
        self.lock = asyncio.Lock()
        self.last_rx = 0
        self.last_tx = 0
        self.last_ping = 0
        self.pings = 0
        self.pid = 0
        self.connects = 0
        self.failures = 0
        self.last_error = None
        self.backoff = 1
        self.hb = time.ticks_ms()
        self.rx = 0
        self.tx = 0

    # -- encoding --

    def _connect_packet(self):
        flags = 0x02  # clean session
        payload = bytearray(_str(self.client_id))
        if self.will:
            topic, msg, retain = self.will
            flags |= 0x04 | (0x20 if retain else 0)
            payload += _str(topic) + _str(msg)
        if self.user:
            flags |= 0x80
            payload += _str(self.user)
            if self.password is not None:
                flags |= 0x40
                payload += _str(self.password)
        var = _str("MQTT") + bytes([4, flags]) + self.keepalive.to_bytes(2, "big")
        return _packet(0x10, var + payload)

    def _next_pid(self):
        self.pid = self.pid % 65535 + 1
        return self.pid

    # -- io --

    async def _send(self, data):
        w = self.w
        if w is None:
            raise OSError("not connected")
        async with self.lock:
            w.write(data)
            await _timeout(w.drain(), WRITE_TIMEOUT)
        self.last_tx = time.ticks_ms()
        self.tx += 1

    async def _read_packet(self):
        r = self.r
        hdr = await r.readexactly(1)
        n = 0
        shift = 0
        while True:
            b = (await r.readexactly(1))[0]
            n |= (b & 0x7F) << shift
            if not b & 0x80:
                break
            shift += 7
            if shift > 21:
                raise OSError("bad remaining length")
        if n > MAX_PACKET:
            # Drain and drop oversized packets without buffering them.
            while n:
                n -= len(await r.read(min(n, 512)))
            self.last_rx = time.ticks_ms()
            return hdr[0], None
        body = await r.readexactly(n) if n else b""
        self.last_rx = time.ticks_ms()
        self.rx += 1
        return hdr[0], body

    async def publish(self, topic, payload, retain=False):
        """QoS 0 publish. Returns False (and drops it) when offline; anything
        that matters is re-published on the next connect."""
        if not self.connected:
            return False
        trace.at(trace.MQTT)
        if isinstance(payload, str):
            payload = payload.encode()
        pkt = _packet(0x30 | (1 if retain else 0), _str(topic) + payload)
        hw.heap_mark("mqtt publish")
        try:
            await self._send(pkt)
            return True
        except Exception as e:
            self._drop("publish failed: %r" % (e,))
            return False

    def _drop(self, why):
        if self.connected:
            log("mqtt: " + why)
        self.last_error = why
        self.connected = False
        # Close the socket now: Stream.close() alone doesn't (see streams).
        stream = self.w or self.r
        self.w = None
        self.r = None
        close_now(stream)

    # -- session --

    async def _connect(self):
        trace.at(trace.MQTT)
        ip = await dns.resolve(self.host)
        self.r, self.w = await _timeout(asyncio.open_connection(ip, self.port), CONNECT_TIMEOUT)
        self.w.write(self._connect_packet())
        await _timeout(self.w.drain(), WRITE_TIMEOUT)
        kind, body = await _timeout(self._read_packet(), CONNECT_TIMEOUT)
        if kind != 0x20 or body is None or len(body) != 2:
            raise OSError("expected CONNACK")
        if body[1] != 0:
            raise OSError("broker refused connection (code %d)" % body[1])
        self.connected = True
        self.last_tx = self.last_ping = time.ticks_ms()
        if self.subs:
            payload = bytearray(self._next_pid().to_bytes(2, "big"))
            for t in self.subs:
                payload += _str(t) + b"\x00"
            await self._send(_packet(0x82, payload))

    async def _reader(self):
        try:
            while True:
                kind, body = await self._read_packet()
                trace.at(trace.MQTT)
                t = kind & 0xF0
                if t == 0x30 and body is not None:
                    self._on_publish(kind, body)
                elif t == 0x90 and body is not None:
                    if b"\x80" in body[2:]:
                        log("mqtt: broker rejected a subscription")
                elif t == 0xD0:
                    pass  # PINGRESP; last_rx already updated
        except Exception as e:
            self._drop("connection lost: %r" % (e,))

    def _on_publish(self, kind, body):
        qos = (kind >> 1) & 3
        retained = bool(kind & 1)
        tlen = int.from_bytes(body[:2], "big")
        topic = bytes(body[2 : 2 + tlen]).decode()
        pos = 2 + tlen
        if qos:
            pid = body[pos : pos + 2]
            pos += 2
            asyncio.create_task(self._ack(pid))
        payload = bytes(body[pos:])
        if self.on_message:
            try:
                self.on_message(topic, payload, retained)
            except Exception as e:
                log("mqtt: handler failed for %s: %r" % (topic, e))

    async def _ack(self, pid):
        try:
            await self._send(b"\x40\x02" + pid)
        except Exception as e:
            self._drop("puback failed: %r" % (e,))

    async def _session(self, wifi):
        reader = asyncio.create_task(self._reader())
        try:
            if self.on_connect:
                await self.on_connect()
            ka = self.keepalive * 1000
            while self.connected:
                trace.at(trace.MQTT)
                self.hb = time.ticks_ms()
                if not wifi.up:
                    self._drop("wifi down")
                    break
                now = time.ticks_ms()
                if time.ticks_diff(now, self.last_rx) > ka * 3 // 2:
                    self._drop("no traffic for %d s, assuming dead" % (ka * 3 // 2000))
                    break
                if time.ticks_diff(now, self.last_ping) >= ka // 2:
                    self.last_ping = now
                    self.pings += 1
                    try:
                        await self._send(b"\xc0\x00")
                    except Exception as e:
                        self._drop("ping failed: %r" % (e,))
                        break
                await asyncio.sleep_ms(250)
        finally:
            if self.connected:
                self._drop("session ended")
            reader.cancel()

    async def run(self, wifi):
        while True:
            self.hb = time.ticks_ms()
            if not wifi.up:
                await asyncio.sleep_ms(500)
                continue
            try:
                await self._connect()
            except Exception as e:
                self.failures += 1
                self.last_error = repr(e)
                self._drop("connect failed")
                log("mqtt: connect to %s:%d failed: %r; retry in %d s" % (self.host, self.port, e, self.backoff))
                await self._wait(self.backoff)
                self.backoff = min(self.backoff * 2, 60)
                continue
            self.backoff = 1
            self.connects += 1
            log("mqtt: connected to %s:%d" % (self.host, self.port))
            try:
                await self._session(wifi)
            except Exception as e:
                log("mqtt: session error: %r" % (e,))
                try:
                    import sys

                    sys.print_exception(e)
                except Exception:
                    pass
                self._drop("session error: %r" % (e,))
            await self._wait(1)

    async def _wait(self, s):
        for _ in range(s * 4):
            self.hb = time.ticks_ms()
            await asyncio.sleep_ms(250)

    def status(self):
        return {
            "connected": self.connected,
            "broker": "%s:%d" % (self.host, self.port),
            "connects": self.connects,
            "failures": self.failures,
            "last_error": self.last_error,
            "rx": self.rx,
            "tx": self.tx,
            "pings": self.pings,
        }
