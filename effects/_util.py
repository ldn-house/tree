"""Shared helpers for effects. MicroPython-safe.

Everything a render() call touches is an int: on MicroPython every float is a
heap object, so float maths inside the per-LED loop allocates thousands of
objects per frame. Precompute per-LED tables here in __init__ (floats and
allocation are fine there) and keep render() to integer arithmetic.

Fixed-point conventions used by the tables:
- ONE = 4096 is 1.0 for heights, radii, phases and angles (a full turn).
- Colour channels are 0-255.
"""

import math
from array import array

ONE = 4096
MASK = ONE - 1


def _wheel(pos):
    if pos < 85:
        return (pos * 3, 255 - pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (255 - pos * 3, 0, pos * 3)
    pos -= 170
    return (0, pos * 3, 255 - pos * 3)


# WHEEL[3*p : 3*p+3] is the RGB rainbow colour at position p (0-255), the
# same colour wheel the original firmware used.
WHEEL = bytearray(768)
for _p in range(256):
    _c = _wheel(_p)
    WHEEL[3 * _p] = _c[0]
    WHEEL[3 * _p + 1] = _c[1]
    WHEEL[3 * _p + 2] = _c[2]

# SIN[p] = round(127.5 + 127.5 * sin(2*pi*p/256)), 0-255.
SIN = bytearray(256)
for _p in range(256):
    SIN[_p] = int(127.5 + 127.5 * math.sin(2 * math.pi * _p / 256) + 0.5)


def fp(x):
    """Float -> ONE-based fixed point int (for use in __init__)."""
    return int(x * ONE + 0.5) if x >= 0 else -int(-x * ONE + 0.5)


def param(ctx, name, default):
    v = ctx.params.get(name)
    return default if v is None else v


class Param:
    """Caches a float param as a fixed-point int, re-reading it only when the
    value object in ctx.params changes (identity check, so no allocation)."""

    def __init__(self, ctx, name, default, scale=ONE):
        self.params = ctx.params
        self.name = name
        self.default = default
        self.scale = scale
        self.raw = None
        self.value = int(default * scale)

    def get(self):
        v = self.params.get(self.name, self.default)
        if v is not self.raw:
            self.raw = v
            self.value = int(v * self.scale)
        return self.value


# Colour params. A param spec (default, 0, 0xFFFFFF, RGB) is a colour: its
# value is a packed 0xRRGGBB int (a small int on MicroPython, so reading it
# never allocates). Tools that only know (default, min, max) still see a
# valid number range.
RGB = "rgb"
RGB_MAX = 0xFFFFFF


def rgb(r, g, b):
    """Pack 0-255 channels into a colour param value."""
    return (int(r) & 255) << 16 | (int(g) & 255) << 8 | (int(b) & 255)


def rgb_spec(r, g, b):
    """PARAMS entry for a colour param defaulting to (r, g, b)."""
    return (rgb(r, g, b), 0, RGB_MAX, RGB)


def is_rgb(spec):
    return len(spec) > 3 and spec[3] == RGB


class ColorParam:
    """A colour param, unpacked into r, g, b when its value changes.
    update() returns True on a change (including the first call)."""

    def __init__(self, ctx, name, default):
        self.params = ctx.params
        self.name = name
        self.default = default
        self.raw = None
        self.value = -1
        self.r = self.g = self.b = 0

    def update(self):
        v = self.params.get(self.name, self.default)
        if v is self.raw:
            return False
        self.raw = v
        if not isinstance(v, int) or not 0 <= v <= RGB_MAX:
            v = self.default
        if v == self.value:
            return False
        self.value = v
        self.r = v >> 16
        self.g = v >> 8 & 255
        self.b = v & 255
        return True


class Phase:
    """Integer phase accumulator: returns 0..ONE-1, advancing one full cycle
    every period_ms at speed 1.0. Accumulating per-frame deltas keeps speed
    changes smooth and keeps every intermediate a small int."""

    def __init__(self, period_ms):
        self.period = int(period_ms)
        self.acc = 0
        self.last = 0

    def update(self, t_ms, speed):
        # speed is fixed point (ONE = 1.0)
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.acc = (self.acc + dt * speed) % (self.period * ONE)
        return self.acc // self.period


class Layout:
    """Per-LED cylindrical tables derived from ctx.coords.

    known[i] is 1 when LED i has a position. For unknown LEDs the tables hold
    index-based stand-ins (h = i/n) so 2D fallbacks keep working.
    - h[i]: height 0..ONE (y in the contract frame).
    - theta[i]: azimuth 0..ONE-1 around the trunk, atan2(z, x) mapped from
      [-pi, pi) onto [0, ONE).
    - r[i]: distance from the trunk axis, ONE = the tree's outer radius
      (98th percentile, so a few stray LEDs don't shrink everything else).
    - x[i], z[i]: horizontal position in the same units as r (signed).

    The tables are computed once per coordinates list and shared by every
    effect, so treat them as read-only. Building them takes float maths per
    LED, which on the Pico means a lot of short-lived heap objects; doing it
    once instead of on every effect switch keeps the auto cycle's heap churn
    near zero.
    """

    def __init__(self, ctx):
        t = _tables(ctx.coords, ctx.n)
        self.n = ctx.n
        self.known, self.h, self.theta, self.r, self.x, self.z, self.count, self.rmax = t


# (coords, n, tables) of the last layout built.
_shared = [None, -1, None]


def _tables(coords, n):
    if _shared[0] is coords and _shared[1] == n:
        return _shared[2]
    _shared[2] = None  # let the old tables go before building new ones
    known = bytearray(n)
    h = array("H", bytes(2 * n))
    theta = array("H", bytes(2 * n))
    r = array("H", bytes(2 * n))
    x = array("h", bytes(2 * n))
    z = array("h", bytes(2 * n))
    # First pass in 1/65536 units so the radius normalisation below is
    # integer maths (fewer float temporaries per LED).
    ri = array("i", bytes(4 * n))
    xi = array("i", bytes(4 * n))
    zi = array("i", bytes(4 * n))
    k_theta = ONE / (2 * math.pi)
    half = ONE / 2
    count = 0
    for i in range(n):
        c = coords[i] if coords and i < len(coords) else None
        if c is None:
            h[i] = i * ONE // n
            continue
        cx, cy, cz = c
        known[i] = 1
        count += 1
        h[i] = min(ONE, max(0, int(cy * ONE + 0.5)))
        theta[i] = int(math.atan2(cz, cx) * k_theta + half) & MASK
        ri[i] = min(1 << 20, int(math.sqrt(cx * cx + cz * cz) * 65536 + 0.5))
        xi[i] = max(-(1 << 20), min(1 << 20, int(cx * 65536)))
        zi[i] = max(-(1 << 20), min(1 << 20, int(cz * 65536)))
    rmax = 65536
    if count:
        radii = sorted(ri[i] for i in range(n) if known[i])
        rmax = radii[int(0.98 * (len(radii) - 1))] or 65536
        radii = None
    for i in range(n):
        if known[i]:
            r[i] = min(65535, (ri[i] * ONE + rmax // 2) // rmax)
            x[i] = max(-32768, min(32767, xi[i] * ONE // rmax))
            z[i] = max(-32768, min(32767, zi[i] * ONE // rmax))
    t = (known, h, theta, r, x, z, count, rmax / 65536)
    _shared[0] = coords
    _shared[1] = n
    _shared[2] = t
    return t


def clear(frame):
    for i in range(len(frame)):
        frame[i] = 0
