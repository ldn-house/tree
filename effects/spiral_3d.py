"""A spiral band wound round the tree, rotating and alternating red/green.

The original comment promised 3 twists but scaled raw y, giving under one
twist on the real coordinates; twists now counts full turns bottom to top.
"""

from effects._util import ONE, MASK, Layout, Param, Phase

NAME = "spiral_3d"
DURATION_MS = 8000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "twists": (3.0, 0.5, 8.0),
    "width": (0.5, 0.1, 1.5),  # half-width of the band, radians
}
NEEDS_COORDS = True

PERIOD_MS = 4000  # one revolution at speed 1
COLOR_SEGMENTS = 6  # colour flips red/green every 1/6 revolution


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.theta = lay.theta
        self.h = lay.h
        self.speed = Param(ctx, "speed", 1.0)
        self.twists = Param(ctx, "twists", 3.0)
        # radians -> fraction of a turn
        self.width = Param(ctx, "width", 0.5, ONE / 6.283185307179586)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        rot = self.phase.update(t_ms, self.speed.get())
        twists = self.twists.get()
        width = self.width.get() or 1
        red = not (rot * COLOR_SEGMENTS >> 12) & 1
        half = ONE >> 1
        known = self.known
        theta = self.theta
        h = self.h
        for i in range(self.n):
            k = 3 * i
            target = (h[i] * twists >> 12) + rot
            d = ((theta[i] - target + half) & MASK) - half
            if d < 0:
                d = -d
            if known[i] and d < width:
                b = 255 - d * 255 // width
                frame[k] = b if red else 0
                frame[k + 1] = 0 if red else b
            else:
                frame[k] = 0
                frame[k + 1] = 0
            frame[k + 2] = 0
