"""Rings expanding outward from the trunk, coloured by height."""

from effects._util import ONE, WHEEL, Layout, Param, Phase

NAME = "radial_burst"
DURATION_MS = 10000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "thickness": (0.15, 0.05, 0.5),
}
NEEDS_COORDS = True

PERIOD_MS = 3300  # one burst at speed 1
REACH = 1.2  # the ring travels out to 1.2x the tree's radius


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.r = lay.r
        # Hue by height, bottom red through the wheel to the top.
        self.hue = bytearray(ctx.n)
        for i in range(ctx.n):
            self.hue[i] = min(255, lay.h[i] >> 4)
        self.speed = Param(ctx, "speed", 1.0)
        self.thickness = Param(ctx, "thickness", 0.15)
        self.phase = Phase(PERIOD_MS)
        self.reach = int(REACH * ONE)

    def render(self, t_ms, frame):
        ring = self.phase.update(t_ms, self.speed.get()) * self.reach >> 12
        thick = self.thickness.get() or 1
        known = self.known
        r = self.r
        hue = self.hue
        for i in range(self.n):
            k = 3 * i
            d = r[i] - ring
            if d < 0:
                d = -d
            if known[i] and d < thick:
                b = 255 - d * 255 // thick
                o = hue[i] * 3
                frame[k] = WHEEL[o] * b >> 8
                frame[k + 1] = WHEEL[o + 1] * b >> 8
                frame[k + 2] = WHEEL[o + 2] * b >> 8
            else:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
