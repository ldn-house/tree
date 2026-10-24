"""Calibration sequence decoding, loaded only when a sequence is used."""

import gc
import struct
import binascii

from app.log import log
from app import seqfile
from app.calib import MAX_SEQUENCE, MIN_FRAME_MS, MONO_LUTS

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

    def _mono_lut(self, levels):
        """One flat bytearray, 256 rows of 24 bytes: row v is 8 LEDs (RGB)
        lit at levels where bit b of v is set. 6 KB per level and a single
        allocation, so expanding a mono frame is 63 slice copies with no
        per-frame garbage. At most MONO_LUTS are kept."""
        lut = self._luts.get(levels)
        if lut is None:
            if len(self._luts) >= MONO_LUTS:
                self._luts.clear()
            gc.collect()
            lut = bytearray(256 * 24)
            lr, lg, lb = levels
            for v in range(256):
                row = 24 * v
                for b in range(8):
                    if v >> b & 1:
                        j = row + 3 * b
                        lut[j] = lr
                        lut[j + 1] = lg
                        lut[j + 2] = lb
            self._luts[levels] = lut
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
            lut = memoryview(self._mono_lut((lr, lg, lb)))
            for j in range(nb):
                o = 24 * j
                r = 24 * d[p + j]
                if o + 24 <= total:
                    out[o : o + 24] = lut[r : r + 24]
                else:
                    out[o:total] = lut[r : r + total - o]
        else:
            # Colour frames: test three bits per LED. No LUT memory at all.
            n = self.n
            pr = p
            pg = p + nb
            pb = p + 2 * nb
            for i in range(n):
                byte = i >> 3
                m = 1 << (i & 7)
                j = 3 * i
                out[j] = lr if d[pr + byte] & m else 0
                out[j + 1] = lg if d[pg + byte] & m else 0
                out[j + 2] = lb if d[pb + byte] & m else 0

    def prewarm(self):
        """Build the mono LUTs playback will need before T0, so none is
        built (slowly, in Python) between two deadlines."""
        d = self.data
        levels = []
        for pos in self.offsets:
            if d[pos + 2] & seqfile.FLAG_MONO:
                lv = (d[pos + 3], d[pos + 4], d[pos + 5])
                if lv not in levels:
                    levels.append(lv)
        if len(levels) > MONO_LUTS:
            log("calibration: %d mono levels; LUTs will be rebuilt during playback" % len(levels))
        for lv in levels[:MONO_LUTS]:
            self._mono_lut(lv)

    def info(self):
        return {"crc32": "%08x" % self.crc, "frames": self.frames, "loop_ms": self.loop_ms}
