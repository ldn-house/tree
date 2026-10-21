"""Calibration sequence files (.tcal), as specified in docs/calibration-protocol.md.

A sequence is a list of frames. Each frame has a hold duration and an RGB
level, and up to three bitplanes saying which LEDs show each channel.

This module is deliberately MicroPython-compatible (struct, binascii only, no
numpy, no f-string-only features) so the firmware can reuse the parser and
iterate frames without loading the whole file into RAM.

Layout (little-endian):

    header  (16 bytes)
        0   4s  magic  b"TCAL"
        4   B   version (1)
        5   B   flags   (reserved, 0)
        6   H   led_count
        8   H   frame_count
        10  H   default_loops (0 = loop until stopped/unlocked)
        12  I   reserved (0)
    frames  (frame_count times)
        0   H   duration_ms (1..65535)
        2   B   frame flags: bit0 MONO (one plane used for all lit channels)
        3   3B  level R, G, B (0-255)
        6   ... planes: 1 (MONO) or 3 (R, G, B) of ceil(led_count/8) bytes;
                LED i is bit (i & 7) of byte (i >> 3), LSB first
    trailer (4 bytes)
        0   I   CRC-32 (binascii.crc32) of every preceding byte
"""

import struct

try:
    import binascii
except ImportError:  # pragma: no cover - MicroPython builds always have it
    import ubinascii as binascii

MAGIC = b"TCAL"
VERSION = 1
HEADER = "<4sBBHHHI"
HEADER_SIZE = 16
FRAME_HEADER = "<HB3B"
FRAME_HEADER_SIZE = 6
FLAG_MONO = 0x01


def plane_bytes(led_count):
    return (led_count + 7) >> 3


def frame_size(led_count, flags):
    planes = 1 if flags & FLAG_MONO else 3
    return FRAME_HEADER_SIZE + planes * plane_bytes(led_count)


def pack_plane(bits, led_count):
    """Pack an iterable of truthy values (one per LED) into a bitplane."""
    out = bytearray(plane_bytes(led_count))
    for i, b in enumerate(bits):
        if i >= led_count:
            break
        if b:
            out[i >> 3] |= 1 << (i & 7)
    return bytes(out)


def unpack_plane(plane, led_count):
    return [(plane[i >> 3] >> (i & 7)) & 1 for i in range(led_count)]


class Frame:
    """One calibration frame.

    planes is a tuple of 1 (mono) or 3 (R, G, B) packed bitplanes.
    """

    def __init__(self, duration_ms, level, planes):
        if not 1 <= duration_ms <= 65535:
            raise ValueError("duration_ms out of range")
        if len(planes) not in (1, 3):
            raise ValueError("need 1 or 3 planes")
        self.duration_ms = duration_ms
        self.level = tuple(level)
        self.planes = tuple(planes)

    @property
    def mono(self):
        return len(self.planes) == 1

    def channel_bits(self, led_count):
        """Return per-channel lists of 0/1 for (R, G, B)."""
        if self.mono:
            bits = unpack_plane(self.planes[0], led_count)
            return [bits if self.level[c] else [0] * led_count for c in range(3)]
        return [unpack_plane(p, led_count) for p in self.planes]

    def rgb(self, led_count):
        """Expand to a bytearray(3 * led_count) in RGB order (reference expander)."""
        out = bytearray(3 * led_count)
        lr, lg, lb = self.level
        if self.mono:
            p = self.planes[0]
            for i in range(led_count):
                if (p[i >> 3] >> (i & 7)) & 1:
                    j = 3 * i
                    out[j] = lr
                    out[j + 1] = lg
                    out[j + 2] = lb
        else:
            pr, pg, pb = self.planes
            for i in range(led_count):
                k = i >> 3
                m = 1 << (i & 7)
                j = 3 * i
                if pr[k] & m:
                    out[j] = lr
                if pg[k] & m:
                    out[j + 1] = lg
                if pb[k] & m:
                    out[j + 2] = lb
        return out


def encode(frames, led_count, default_loops=1):
    """Serialise frames to .tcal bytes."""
    n = plane_bytes(led_count)
    body = bytearray(struct.pack(HEADER, MAGIC, VERSION, 0, led_count,
                                 len(frames), default_loops, 0))
    for f in frames:
        flags = FLAG_MONO if f.mono else 0
        body += struct.pack(FRAME_HEADER, f.duration_ms, flags, *f.level)
        for p in f.planes:
            if len(p) != n:
                raise ValueError("plane has wrong length")
            body += p
    body += struct.pack("<I", binascii.crc32(bytes(body)) & 0xFFFFFFFF)
    return bytes(body)


def decode(data):
    """Parse .tcal bytes. Returns (led_count, default_loops, [Frame, ...])."""
    if len(data) < HEADER_SIZE + 4:
        raise ValueError("truncated")
    magic, version, _flags, led_count, frame_count, loops, _r = struct.unpack(
        HEADER, data[:HEADER_SIZE])
    if magic != MAGIC:
        raise ValueError("bad magic")
    if version != VERSION:
        raise ValueError("unsupported version")
    (crc,) = struct.unpack("<I", data[-4:])
    if binascii.crc32(data[:-4]) & 0xFFFFFFFF != crc:
        raise ValueError("bad crc")
    n = plane_bytes(led_count)
    frames = []
    pos = HEADER_SIZE
    for _ in range(frame_count):
        duration, flags, lr, lg, lb = struct.unpack(
            FRAME_HEADER, data[pos:pos + FRAME_HEADER_SIZE])
        pos += FRAME_HEADER_SIZE
        count = 1 if flags & FLAG_MONO else 3
        planes = []
        for _p in range(count):
            planes.append(bytes(data[pos:pos + n]))
            pos += n
        frames.append(Frame(duration, (lr, lg, lb), planes))
    if pos != len(data) - 4:
        raise ValueError("length mismatch")
    return led_count, loops, frames


def frame_offsets(data):
    """Yield (byte_offset, duration_ms) per frame without copying planes.

    Lets a player keep the file in flash or a single buffer and expand frames
    lazily.
    """
    _m, _v, _f, led_count, frame_count, _l, _r = struct.unpack(
        HEADER, data[:HEADER_SIZE])
    pos = HEADER_SIZE
    for _ in range(frame_count):
        duration, flags = struct.unpack("<HB", data[pos:pos + 3])
        yield pos, duration
        pos += frame_size(led_count, flags)
