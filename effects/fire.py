"""Fire burning up the tree.

A heat field in the spirit of Fire2012, but laid out in 3D: heat is fed at
the base and cools with height (the real y coordinate). Each column of the
tree (its angle round the trunk) has its own flame height, which wanders
with slow noise, and turbulence noise scrolls upwards through the field so
flame tongues rise and break up as they climb. Now and then a column's
flame licks well above the rest. Every noise layer wraps in time, so the
animation repeats seamlessly every PERIOD_MS.

Palette: black, deep red, orange, and yellow at the hottest.
"""

from array import array

from effects._util import ONE, Layout, Param, Phase

NAME = "fire"
DURATION_MS = 30000
PARAMS = {
    "speed": (1.0, 0.3, 3.0),
    "height": (1.0, 0.5, 1.5),  # flame height
}
NEEDS_COORDS = True

PERIOD_MS = 10000  # the whole fire repeats after this at speed 1

# Value-noise lattices: COLS cells round the trunk, ROWS cells high (the
# turbulence lattice wraps vertically so it can scroll for ever).
COLS = 8
ROWS = 16  # a power of two
STRIDE = ROWS + 1
CELLS_UP = 3  # turbulence lattice cells per tree height
SCROLL = (2, 3)  # lattice heights each turbulence layer rises per period
SLOW = 12  # time cells per period in the flame-height noise
FLAME = 0.55  # mean flame height, tree heights
WANDER = 0.14  # flame height variation either way
LICK = 0.35  # extra height of a lick at its peak
LICK_FROM = 170  # lick noise (0-255) above this raises the flame
HEAT = 300  # heat at the base before turbulence (255 = full)

# Heat 0-255 -> colour, linear between these stops.
STOPS = (
    (0, (0, 0, 0)),
    (50, (40, 1, 0)),
    (110, (150, 10, 0)),
    (170, (230, 50, 0)),
    (215, (255, 115, 0)),
    (255, (255, 165, 15)),
)


def _palette():
    pal = bytearray(768)
    for k in range(len(STOPS) - 1):
        h0, c0 = STOPS[k]
        h1, c1 = STOPS[k + 1]
        for h in range(h0, h1 + 1):
            f = (h - h0) / (h1 - h0)
            for ch in range(3):
                pal[3 * h + ch] = int(c0[ch] + (c1[ch] - c0[ch]) * f + 0.5)
    return pal


PALETTE = _palette()

# Smoothstep weights: SMOOTH[f] for f in 0..255, 0..256.
SMOOTH = array("H", (int(256 * (3 * (f / 256) ** 2 - 2 * (f / 256) ** 3) + 0.5) for f in range(256)))


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        rand = ctx.random.getrandbits
        lay = Layout(ctx)
        # Per LED: its two neighbouring lattice columns and the blend
        # between them (static, since angles don't move).
        self.c0 = bytearray(n)
        self.c1 = bytearray(n)
        self.cw = bytearray(n)
        # Height in turbulence-lattice units of 1/256 cell.
        self.hy = array("H", bytes(2 * n))
        # Height for the flame test; unplaced LEDs sit above any flame.
        self.h = array("H", lay.h)
        for i in range(n):
            a = lay.theta[i] * COLS * 256 // ONE
            c = a >> 8
            self.c0[i] = c % COLS
            self.c1[i] = (c + 1) % COLS
            self.cw[i] = a & 255
            self.hy[i] = lay.h[i] * CELLS_UP * 256 // ONE
            if not lay.known[i]:
                self.h[i] = 0xFFFF
        # Two turbulence lattices (COLS x ROWS, wrapping both ways), already
        # blended across columns for each LED's angle, so render() only
        # interpolates vertically: ta[STRIDE * i + row], with row 0 repeated
        # at the end so row + 1 never needs wrapping. And the slow lattices
        # for flame height and licks (COLS x SLOW).
        lattice = bytearray(rand(8) for _ in range(2 * COLS * ROWS))
        self.ta = bytearray(STRIDE * n)
        self.tb = bytearray(STRIDE * n)
        for i in range(n):
            w = SMOOTH[self.cw[i]]
            for row in range(STRIDE):
                for t, off in ((self.ta, 0), (self.tb, COLS * ROWS)):
                    p = lattice[off + row % ROWS * COLS + self.c0[i]]
                    q = lattice[off + row % ROWS * COLS + self.c1[i]]
                    t[STRIDE * i + row] = p + ((q - p) * w >> 8)
        lattice = None
        self.wander = bytearray(rand(8) for _ in range(COLS * SLOW))
        self.lick = bytearray(rand(8) for _ in range(COLS * SLOW))
        # Per-column flame height for the current frame, in h units.
        self.top = array("H", bytes(2 * COLS))
        self.mean = int(FLAME * ONE)
        self.amp = int(WANDER * ONE)
        self.lk = int(LICK * ONE)
        self.speed = Param(ctx, "speed", 1.0)
        self.height = Param(ctx, "height", 1.0, 256)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        ph = self.phase.update(t_ms, self.speed.get())
        scale = self.height.get()

        # Flame height per column from the slow lattices, sampled at this
        # point in the period (wrapping in time).
        s = ph * SLOW * 256 // ONE
        s0 = s >> 8
        sw = SMOOTH[s & 255]
        s1 = (s0 + 1) % SLOW
        s0 = s0 % SLOW
        wander = self.wander
        lick = self.lick
        top = self.top
        mean = self.mean
        amp = self.amp
        lk = self.lk
        for c in range(COLS):
            w = wander[s0 * COLS + c]
            w += (wander[s1 * COLS + c] - w) * sw >> 8
            v = lick[s0 * COLS + c]
            v += (lick[s1 * COLS + c] - v) * sw >> 8
            f = mean + (w - 128) * amp // 128
            if v > LICK_FROM:
                q = (v - LICK_FROM) * 256 // (255 - LICK_FROM)
                f += lk * q * q >> 16
            top[c] = f * scale >> 8

        # Turbulence scroll offsets (1/256 lattice cell), wrapping exactly
        # once per SCROLL[k] lattice heights per period.
        span = ROWS * 256
        off0 = ph * SCROLL[0] * span // ONE
        off1 = ph * SCROLL[1] * span // ONE
        wrap = span - 1
        ta = self.ta
        tb = self.tb
        h = self.h
        hy = self.hy
        c0 = self.c0
        c1 = self.c1
        cw = self.cw
        pal = PALETTE
        smooth = SMOOTH
        for i in range(self.n):
            k = 3 * i
            # Column flame height, blended between the two columns.
            f = top[c0[i]]
            f += (top[c1[i]] - f) * smooth[cw[i]] >> 8
            hi = h[i]
            if hi >= f:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            # d: 256 at the base down to 0 at the flame's tip.
            d = (f - hi) * 256 // f
            # Turbulence: two layers rising at different speeds; the field
            # moves down through the lattice, so features climb the tree.
            y = (hy[i] - off0) & wrap
            r = STRIDE * i + (y >> 8)
            p = ta[r]
            n0 = p + ((ta[r + 1] - p) * smooth[y & 255] >> 8)
            y = (hy[i] * 2 - off1) & wrap
            r = STRIDE * i + (y >> 8)
            p = tb[r]
            n1 = p + ((tb[r + 1] - p) * smooth[y & 255] >> 8)
            # Base heat cools from the bottom to the tip; turbulence matters
            # more towards the tip, so the base stays a solid bed of embers
            # and the flames break up as they rise.
            m = 256 + (((n0 + n1) - 255) * (406 - d) >> 8)
            heat = d * HEAT * m >> 16
            if heat > 255:
                heat = 255
            elif heat < 0:
                heat = 0
            heat *= 3
            frame[k] = pal[heat]
            frame[k + 1] = pal[heat + 1]
            frame[k + 2] = pal[heat + 2]
