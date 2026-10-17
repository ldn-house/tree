"""Calibration session and sequence player (docs/calibration-protocol.md).

The firmware is a timed player of host-built .tcal frames. While a session
holds the lock the effect engine is paused and every calibration frame goes
through the output stage with brightness 255 and gamma off, so the power cap
still applies (review M6). The lock has a TTL (review M7), MQTT keeps running
throughout, and the tree always boots idle.

Playback schedules each frame's latch against absolute ticks_us deadlines:
the DMA transfer starts write_us before the deadline, so the strip latches on
time. The next frame is expanded and run through the output stage during the
current frame's hold.
"""

import gc
import struct
import time

import asyncio

try:
    import binascii
except ImportError:  # pragma: no cover
    import ubinascii as binascii

import bootmgr

from app import seqfile
from app.httpd import HTTPError
from app.log import log

PROTOCOL = 1
TTL_DEFAULT_S = 180
TTL_MAX_S = 900
MIN_FRAME_MS = 20
MAX_SEQUENCE = 65536
MAX_DELAY_MS = 30000
IMPLICIT_HOLD_MS = 2000
STATIC_REFRESH_MS = 1000
SPIN_US = 2500
LUT_CACHE = 8
GC_SLACK_US = 12000
MIN_LEAD_MS = 10
LEGACY_LED = (255, 255, 255)
LEGACY_BINARY = (80, 80, 80)

_POP = bytes(bin(i).count("1") for i in range(256))


class Sequence:
    """A validated .tcal sequence held in RAM, expanded lazily per frame."""

    def __init__(self, data, n, budget, check_durations=True):
        if len(data) > MAX_SEQUENCE:
            raise ValueError("sequence larger than %d bytes" % MAX_SEQUENCE)
        if len(data) < seqfile.HEADER_SIZE + 4:
            raise ValueError("truncated")
        magic, version, flags, count, frames, loops, _r = struct.unpack(seqfile.HEADER, data[: seqfile.HEADER_SIZE])
        if magic != seqfile.MAGIC:
            raise ValueError("bad magic")
        if version != seqfile.VERSION:
            raise ValueError("unsupported version %d" % version)
        if flags:
            raise ValueError("reserved header flags set")
        if count != n:
            raise ValueError("led_count %d, tree has %d" % (count, n))
        if frames < 1:
            raise ValueError("no frames")
        (crc,) = struct.unpack("<I", data[-4:])
        if binascii.crc32(data[:-4]) & 0xFFFFFFFF != crc:
            raise ValueError("bad crc")
        self.data = data
        self.n = n
        self.crc = crc
        self.default_loops = loops
        self.offsets = []
        self.durations = []
        self.over_budget = []
        nb = seqfile.plane_bytes(n)
        pos = seqfile.HEADER_SIZE
        for k in range(frames):
            if pos + seqfile.FRAME_HEADER_SIZE > len(data) - 4:
                raise ValueError("length mismatch")
            dur, fflags = struct.unpack("<HB", data[pos : pos + 3])
            if fflags & ~seqfile.FLAG_MONO:
                raise ValueError("frame %d: reserved flags set" % k)
            if check_durations and dur < MIN_FRAME_MS:
                raise ValueError("frame %d: duration %d ms below min_frame_ms %d" % (k, dur, MIN_FRAME_MS))
            if dur < 1:
                raise ValueError("frame %d: zero duration" % k)
            self.offsets.append(pos)
            self.durations.append(dur)
            if self._raw_sum(pos, fflags, nb) > budget:
                self.over_budget.append(k)
            pos += seqfile.frame_size(n, fflags)
        if pos != len(data) - 4:
            raise ValueError("length mismatch")
        self.frames = frames
        self.loop_ms = sum(self.durations)
        self._luts = {}

    def _count(self, start, nb):
        d = self.data
        c = 0
        for j in range(start, start + nb - 1):
            c += _POP[d[j]]
        last = d[start + nb - 1]
        if self.n & 7:
            last &= (1 << (self.n & 7)) - 1
        return c + _POP[last]

    def _raw_sum(self, pos, fflags, nb):
        lv = self.data[pos + 3 : pos + 6]
        p = pos + seqfile.FRAME_HEADER_SIZE
        if fflags & seqfile.FLAG_MONO:
            return self._count(p, nb) * (lv[0] + lv[1] + lv[2])
        return sum(self._count(p + c * nb, nb) * lv[c] for c in range(3))

    def _lut(self, levels):
        """LUT[byte] -> 24-byte chunk (8 LEDs x RGB) as a big-endian int,
        with channel c at levels[c] for each set bit."""
        lut = self._luts.get(levels)
        if lut is None:
            if len(self._luts) >= 2 * LUT_CACHE:
                self._luts.clear()
            lut = []
            for v in range(256):
                x = 0
                for b in range(8):
                    if v >> b & 1:
                        for c in range(3):
                            if levels[c]:
                                x |= levels[c] << ((23 - 3 * b - c) * 8)
                lut.append(x)
            self._luts[levels] = lut
        return lut

    def _mono_lut(self, levels):
        """Like _lut but as 24-byte chunks, so mono frames (most of a
        standard sequence) expand with slice copies and no allocation."""
        key = ("b",) + levels
        lut = self._luts.get(key)
        if lut is None:
            lut = [x.to_bytes(24, "big") for x in self._lut(levels)]
            self._luts[key] = lut
        return lut

    def expand(self, k, out):
        """Write frame k as logical RGB into out (bytearray(3 * n))."""
        d = self.data
        pos = self.offsets[k]
        fflags = d[pos + 2]
        lr, lg, lb = d[pos + 3], d[pos + 4], d[pos + 5]
        nb = seqfile.plane_bytes(self.n)
        p = pos + seqfile.FRAME_HEADER_SIZE
        total = 3 * self.n
        if fflags & seqfile.FLAG_MONO:
            lut = self._mono_lut((lr, lg, lb))
            for j in range(nb):
                o = 24 * j
                chunk = lut[d[p + j]]
                if o + 24 <= total:
                    out[o : o + 24] = chunk
                else:
                    out[o:total] = chunk[: total - o]
        else:
            lr_ = self._lut((lr, 0, 0))
            lg_ = self._lut((0, lg, 0))
            lb_ = self._lut((0, 0, lb))
            for j in range(nb):
                o = 24 * j
                chunk = (lr_[d[p + j]] | lg_[d[p + nb + j]] | lb_[d[p + 2 * nb + j]]).to_bytes(24, "big")
                if o + 24 <= total:
                    out[o : o + 24] = chunk
                else:
                    out[o:total] = chunk[: total - o]

    def prewarm(self):
        """Build every LUT playback will need, so none is built (slowly, in
        Python) between two deadlines."""
        d = self.data
        for pos in self.offsets:
            lv = (d[pos + 3], d[pos + 4], d[pos + 5])
            if d[pos + 2] & seqfile.FLAG_MONO:
                self._mono_lut(lv)
            else:
                self._lut((lv[0], 0, 0))
                self._lut((0, lv[1], 0))
                self._lut((0, 0, lv[2]))
        if len(self._luts) > LUT_CACHE:
            log("calibration: %d distinct levels; LUTs will be rebuilt during playback" % len(self._luts))

    def info(self):
        return {"crc32": "%08x" % self.crc, "frames": self.frames, "loop_ms": self.loop_ms}


class Log:
    def __init__(self, name):
        self.sequence = name
        self.shown = 0
        self.skipped = 0
        self.late_max = 0
        self.late_sum = 0
        self.over_4ms = 0
        self.capped = 0
        self.by_frame = []
        self.worst = None  # (loop, frame) of the latest latch

    def record(self, late_us, skipped=False, where=None):
        if skipped:
            self.skipped += 1
        else:
            self.shown += 1
            self.late_sum += late_us
            if late_us > self.late_max:
                self.late_max = late_us
                self.worst = where
            if late_us > 4000:
                self.over_4ms += 1
        if len(self.by_frame) < 600:
            self.by_frame.append(-1 if skipped else late_us)

    def new_loop(self):
        self.by_frame = []

    def payload(self):
        return {
            "sequence": self.sequence,
            "frames_shown": self.shown,
            "frames_skipped": self.skipped,
            "late_us": {
                "max": self.late_max,
                "mean": self.late_sum // self.shown if self.shown else 0,
                "over_4ms": self.over_4ms,
            },
            "capped_frames": self.capped,
            "worst_frame": self.worst,
            "late_us_by_frame": self.by_frame,
        }


class Calib:
    def __init__(self, rt):
        self.rt = rt
        self.n = rt.n
        self.state = "idle"
        self.expires = 0
        self.ttl_s = TTL_DEFAULT_S
        self.implicit = False
        self.static = bytearray(3 * self.n)
        self.static_dirty = False
        self.static_at = 0
        self.legacy_led = -1
        self.uploaded = None
        self._default = None
        self.player = None
        self.play_info = None
        self.log = None
        self.version = 0
        self.published_at = 0
        self.hb = time.ticks_ms()
        self.bufs = (bytearray(3 * self.n), bytearray(3 * self.n))
        self.raw = bytearray(3 * self.n)

    # -- state --

    def playing(self):
        return self.state == "playing"

    def locked(self):
        return self.state != "idle"

    def _changed(self):
        self.version += 1

    def ttl_left(self):
        if self.state == "idle":
            return 0
        return max(0, time.ticks_diff(self.expires, time.ticks_ms()) // 1000)

    def refresh(self, ttl_s=None):
        if ttl_s is not None:
            self.ttl_s = min(TTL_MAX_S, max(1, ttl_s))
        if self.implicit and self.state == "locked":
            return  # an implicit lock ends 2 s after playback, polls or not
        if self.state != "idle":
            t = time.ticks_add(time.ticks_ms(), self.ttl_s * 1000)
            if time.ticks_diff(t, self.expires) > 0:
                self.expires = t

    def lock(self, ttl_s=None, implicit=False):
        if self.state == "idle":
            self.ttl_s = TTL_DEFAULT_S
            self.state = "locked"
            self.implicit = implicit
            self.expires = time.ticks_ms()
            self.rt.engine.owner = self
            self._clear()
            self.legacy_led = -1
            log("calibration: locked")
            self._changed()
        elif not implicit:
            self.implicit = False
        self.refresh(ttl_s if ttl_s is not None else self.ttl_s)

    def unlock(self, why):
        if self.state == "idle":
            return
        self._stop_player()
        self.state = "idle"
        self.implicit = False
        self._default = None
        self.legacy_led = -1
        # All off for at least one frame, then hand the strip back to the
        # effect engine, which shows the (possibly updated) user state.
        self._clear()
        self.rt.engine.owner = self
        asyncio.create_task(self._release())
        log("calibration: unlocked (%s)" % why)
        self._changed()

    async def _release(self):
        await self._show_static()
        await asyncio.sleep_ms(40)
        if self.state == "idle":
            self.rt.engine.owner = None

    def _clear(self):
        s = self.static
        for i in range(len(s)):
            s[i] = 0
        self.static_dirty = True

    def _set_static(self, rgb_fn):
        self._clear()
        s = self.static
        for i in range(self.n):
            c = rgb_fn(i)
            if c:
                s[3 * i] = c[0]
                s[3 * i + 1] = c[1]
                s[3 * i + 2] = c[2]
        self.static_dirty = True

    async def _show_static(self):
        await self.rt.engine.present(self.static, self.rt.cal_out, 255)
        self.static_dirty = False
        self.static_at = time.ticks_ms()

    async def run(self):
        while True:
            now = time.ticks_ms()
            self.hb = now
            if self.state != "idle":
                # Finite playback pushed expires past its end (play()), so
                # this only cuts infinite loops or an idle lock.
                if time.ticks_diff(now, self.expires) >= 0:
                    self.unlock("playback finished" if self.implicit and not self.playing() else "ttl expired")
                elif self.state == "locked" and (
                    self.static_dirty or time.ticks_diff(now, self.static_at) >= STATIC_REFRESH_MS
                ):
                    await self._show_static()
            await asyncio.sleep_ms(20 if self.static_dirty else 100)

    # -- sequences --

    def default_sequence(self):
        if self._default is None:
            data = None
            for path in (self.rt.slot_dir + "/data/calib_default.tcal", "calib_default.tcal"):
                try:
                    with open(path, "rb") as f:
                        data = f.read(MAX_SEQUENCE + 1)
                    break
                except OSError:
                    continue
            if data is None:
                return None
            self._default = Sequence(data, self.n, self.rt.cal_out.budget)
        return self._default

    def _get_sequence(self, name):
        if name == "uploaded":
            if self.uploaded is None:
                raise HTTPError(409, "no uploaded sequence")
            return self.uploaded
        if name == "default":
            try:
                seq = self.default_sequence()
            except ValueError as e:
                raise HTTPError(409, "default sequence is invalid: %s" % e)
            if seq is None:
                raise HTTPError(409, "no default sequence")
            return seq
        raise HTTPError(400, "sequence must be uploaded or default")

    # -- playback --

    def play(self, name, loops=None, delay_ms=0, ttl_s=None):
        if self.playing():
            raise HTTPError(409, "already playing")
        seq = self._get_sequence(name)
        if loops is None:
            loops = seq.default_loops
        if not 0 <= loops <= 65535:
            raise HTTPError(400, "loops out of range")
        if not 0 <= delay_ms <= MAX_DELAY_MS:
            raise HTTPError(400, "delay_ms out of range")
        implicit = self.state == "idle"
        self.lock(ttl_s, implicit=implicit)
        seq.prewarm()
        # The first latch needs the frame on the wire first: T0 can't be
        # sooner than one write (plus margin) after the request.
        delay_ms = max(delay_ms, self.rt.driver.write_us // 1000 + MIN_LEAD_MS)
        t0 = time.ticks_add(time.ticks_us(), delay_ms * 1000)
        self.state = "playing"
        self.log = Log(name)
        self.play_info = {"sequence": name, "loop": 0, "loops": loops, "frame": 0, "t0_us": t0, "loop_ms": seq.loop_ms}
        if loops:
            end = time.ticks_add(time.ticks_ms(), delay_ms + loops * seq.loop_ms + IMPLICIT_HOLD_MS)
            if time.ticks_diff(end, self.expires) > 0:
                self.expires = end
        self.player = asyncio.create_task(self._play(seq, loops, t0))
        self._changed()
        log("calibration: playing %s, %d loops, t0 in %d ms" % (name, loops, delay_ms))
        return {"sequence": name, "loops": loops, "loop_ms": seq.loop_ms, "t0_in_ms": delay_ms}

    def halt(self):
        if self.state == "idle":
            raise HTTPError(409, "not locked")
        self._stop_player()
        self._clear()
        self._changed()

    def _stop_player(self):
        if self.player is not None:
            self.player.cancel()
            self.player = None
        if self.state == "playing":
            self.state = "locked"
        self.play_info = None

    async def _wait_until(self, kick):
        while True:
            d = time.ticks_diff(kick, time.ticks_us())
            if d > SPIN_US + 1000:
                self.hb = time.ticks_ms()
                await asyncio.sleep_ms((d - SPIN_US) // 1000)
                continue
            while time.ticks_diff(kick, time.ticks_us()) > 0:
                pass
            return

    async def _play(self, seq, loops, t0):
        drv = self.rt.driver
        out = self.rt.cal_out
        write_us = drv.write_us
        bufs = self.bufs
        raw = self.raw
        info = self.play_info
        lg = self.log
        k = 0
        loop = 0
        bi = 0
        due = t0
        seq.expand(0, raw)
        out.process(raw, bufs[0], 255)
        capped = out.capped
        try:
            while loops == 0 or loop < loops:
                kick = time.ticks_add(due, -write_us)
                await self._wait_until(kick)
                while drv.busy():
                    pass
                late = time.ticks_diff(time.ticks_us(), kick)
                dur = seq.durations[k] * 1000
                if late > dur:
                    lg.record(late, skipped=True)
                else:
                    drv.write(bufs[bi])
                    lg.record(late, where=(loop, k))
                    if capped:
                        lg.capped += 1
                info["frame"] = k
                info["loop"] = loop
                due = time.ticks_add(due, dur)
                k += 1
                if k == seq.frames:
                    k = 0
                    loop += 1
                    if loops == 0 or loop < loops:
                        lg.new_loop()
                    if loops and loop >= loops:
                        break
                bi ^= 1
                seq.expand(k, raw)
                out.process(raw, bufs[bi], 255)
                capped = out.capped
                # Collect garbage now, in the hold, rather than letting an
                # automatic collection land on a deadline.
                if time.ticks_diff(time.ticks_add(due, -write_us), time.ticks_us()) > GC_SLACK_US:
                    gc.collect()
                self.hb = time.ticks_ms()
                await asyncio.sleep_ms(0)
            # Hold the last frame for its duration, then all off.
            await self._wait_until(due)
            self.player = None
            self.state = "locked"
            self.play_info = None
            self._clear()
            if self.implicit:
                self.expires = time.ticks_add(time.ticks_ms(), IMPLICIT_HOLD_MS)
            self._changed()
            log("calibration: playback done, %d shown, %d skipped, max late %d us" % (lg.shown, lg.skipped, lg.late_max))
        except asyncio.CancelledError:
            pass

    # -- payloads --

    def status(self):
        uploaded = self.uploaded.info() if self.uploaded else None
        default = None
        try:
            seq = self.default_sequence()
            default = seq.info() if seq else None
        except ValueError:
            default = None
        playing = None
        if self.playing() and self.play_info:
            p = self.play_info
            playing = {
                "sequence": p["sequence"],
                "loop": p["loop"],
                "loops": p["loops"],
                "frame": p["frame"],
                "t0_ms_ago": time.ticks_diff(time.ticks_us(), p["t0_us"]) // 1000,
            }
        if self.state == "idle":
            self._default = None
        return {
            "protocol": PROTOCOL,
            "state": self.state,
            "led_count": self.n,
            "power_budget": self.rt.cal_out.budget,
            "min_frame_ms": MIN_FRAME_MS,
            "max_sequence_bytes": MAX_SEQUENCE,
            "ttl_s": self.ttl_left(),
            "sequences": {"default": default, "uploaded": uploaded},
            "playing": playing,
        }

    def state_payload(self):
        p = self.play_info or {}
        return {
            "state": self.state,
            "sequence": p.get("sequence"),
            "loop": p.get("loop"),
            "loops": p.get("loops"),
            "loop_ms": p.get("loop_ms"),
            "ttl_s": self.ttl_left(),
        }

    def mqtt_command(self, cmd):
        action = cmd.get("action")
        try:
            if action == "play":
                loops = cmd.get("loops")
                delay = cmd.get("delay_ms", 0)
                ttl = cmd.get("ttl_s")
                for v in (loops, delay, ttl):
                    if v is not None and (not isinstance(v, int) or isinstance(v, bool)):
                        return "loops, delay_ms and ttl_s must be integers"
                self.play(cmd.get("sequence", "default"), loops, delay, ttl)
            elif action == "halt":
                self.halt()
            elif action == "unlock":
                self.unlock("mqtt unlock")
            else:
                return "unknown action %r" % (action,)
        except HTTPError as e:
            return e.msg
        self.refresh()
        return None

    # -- HTTP --

    def routes(self, srv):
        srv.route("GET", "/calibrate", self.h_status)
        srv.route("GET", "/calibrate/status", self.h_legacy_status)
        srv.route("GET", "/calibrate/log", self.h_log)
        for path, h in (
            ("/calibrate/start", self.h_start),
            ("/calibrate/stop", self.h_stop),
            ("/calibrate/led", self.h_led),
            ("/calibrate/binary", self.h_binary),
            ("/calibrate/clear", self.h_clear),
        ):
            srv.route("GET|POST", path, h, auth=True)
        srv.route("POST", "/calibrate/sequence", self.h_sequence, auth=True)
        srv.route("POST", "/calibrate/play", self.h_play, auth=True)
        srv.route("POST", "/calibrate/halt", self.h_halt, auth=True)
        srv.route("POST", "/calibrate/frame", self.h_frame, auth=True)

    async def h_status(self, req):
        self.refresh()
        return 200, self.status()

    async def h_legacy_status(self, req):
        self.refresh()
        return 200, {"calibration_mode": self.locked(), "current_led": self.legacy_led, "total_leds": self.n}

    async def h_log(self, req):
        self.refresh()
        if self.log is None:
            return 200, {"sequence": None, "frames_shown": 0, "frames_skipped": 0,
                         "late_us": {"max": 0, "mean": 0, "over_4ms": 0}, "capped_frames": 0}
        return 200, self.log.payload()

    async def h_start(self, req):
        self.lock(req.qint("ttl_s", None, 1, TTL_MAX_S))
        return 200, {"success": True, "mode": "calibration", "state": self.state, "ttl_s": self.ttl_left()}

    async def h_stop(self, req):
        self.unlock("http stop")
        return 200, {"success": True, "mode": "normal", "state": self.state}

    def _need_static(self):
        if self.state == "idle":
            raise HTTPError(400, "Not in calibration mode. Call /calibrate/start first")
        if self.playing():
            raise HTTPError(409, "playing; halt first")
        self.refresh()

    async def h_led(self, req):
        self._need_static()
        n = req.qint("n", -1, -1, self.n - 1)
        self.legacy_led = n
        self._set_static(lambda i: LEGACY_LED if i == n else None)
        return 200, {"success": True, "led": n}

    async def h_binary(self, req):
        self._need_static()
        bit = req.qint("bit", 0, 0, 15)
        self.legacy_led = -2
        self._set_static(lambda i: LEGACY_BINARY if (i >> bit) & 1 else None)
        on = sum(1 for i in range(self.n) if (i >> bit) & 1)
        return 200, {"success": True, "bit": bit, "leds_on": on}

    async def h_clear(self, req):
        self._need_static()
        self.legacy_led = -1
        self._clear()
        return 200, {"success": True}

    async def h_sequence(self, req):
        if self.playing():
            raise HTTPError(409, "cannot replace the sequence while playing")
        data = await req.body(MAX_SEQUENCE)
        try:
            seq = Sequence(data, self.n, self.rt.cal_out.budget)
        except ValueError as e:
            raise HTTPError(400, str(e))
        self.uploaded = seq
        self.refresh()
        out = seq.info()
        out["over_budget_frames"] = seq.over_budget
        return 200, out

    async def h_play(self, req):
        res = self.play(
            req.query.get("sequence") or ("uploaded" if self.uploaded else "default"),
            req.qint("loops", None, 0, 65535),
            req.qint("delay_ms", 0, 0, MAX_DELAY_MS),
            req.qint("ttl_s", None, 1, TTL_MAX_S),
        )
        return 200, res

    async def h_halt(self, req):
        self.halt()
        self.refresh()
        return 200, {"state": self.state}

    async def h_frame(self, req):
        if self.state == "idle":
            raise HTTPError(409, "not locked")
        if self.playing():
            raise HTTPError(409, "playing; halt first")
        data = await req.body(4096)
        try:
            seq = Sequence(data, self.n, self.rt.cal_out.budget, check_durations=False)
        except ValueError as e:
            raise HTTPError(400, str(e))
        if seq.frames != 1:
            raise HTTPError(400, "frame_count must be 1")
        seq.expand(0, self.static)
        self.static_dirty = True
        self.legacy_led = -1
        self.refresh()
        return 200, {"over_budget": bool(seq.over_budget)}
