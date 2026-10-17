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
    """

    def __init__(self, ctx):
        n = ctx.n
        coords = ctx.coords
        self.n = n
        self.known = bytearray(n)
        self.h = array("H", bytes(2 * n))
        self.theta = array("H", bytes(2 * n))
        self.r = array("H", bytes(2 * n))
        self.x = array("h", bytes(2 * n))
        self.z = array("h", bytes(2 * n))
        self.count = 0
        radii = []
        if coords:
            for i in range(n):
                c = coords[i] if i < len(coords) else None
                if c is not None:
                    radii.append(math.sqrt(c[0] * c[0] + c[2] * c[2]))
        rmax = 1.0
        if radii:
            radii.sort()
            rmax = radii[int(0.98 * (len(radii) - 1))] or 1.0
        self.rmax = rmax
        tau = 2 * math.pi
        for i in range(n):
            c = coords[i] if coords and i < len(coords) else None
            if c is None:
                self.h[i] = i * ONE // n
                continue
            x, y, z = c
            self.known[i] = 1
            self.count += 1
            self.h[i] = min(ONE, max(0, fp(y)))
            self.theta[i] = int((math.atan2(z, x) + math.pi) / tau * ONE) & MASK
            self.r[i] = min(65535, fp(math.sqrt(x * x + z * z) / rmax))
            self.x[i] = max(-32768, min(32767, fp(x / rmax)))
            self.z[i] = max(-32768, min(32767, fp(z / rmax)))


def clear(frame):
    for i in range(len(frame)):
        frame[i] = 0
