"""A spiral band wound round the tree, rotating and alternating red/green.

The two colours are params, so the F1 automation can wind it in the
winning team's colour and white.

The original comment promised 3 twists but scaled raw y, giving under one
twist on the real coordinates; twists now counts full turns bottom to top.
"""

from effects._util import ONE, MASK, ColorParam, Layout, Param, Phase, rgb_spec

NAME = "spiral_3d"
DURATION_MS = 8000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "twists": (3.0, 0.5, 8.0),
    "width": (0.5, 0.1, 1.5),  # half-width of the band, radians
    "color": rgb_spec(255, 0, 0),
    "color2": rgb_spec(0, 255, 0),
}
NEEDS_COORDS = True

PERIOD_MS = 4000  # one revolution at speed 1
COLOR_SEGMENTS = 6  # colour flips between the two every 1/6 revolution


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
        self.c1 = ColorParam(ctx, "color", PARAMS["color"][0])
        self.c2 = ColorParam(ctx, "color2", PARAMS["color2"][0])

    def render(self, t_ms, frame):
        rot = self.phase.update(t_ms, self.speed.get())
        twists = self.twists.get()
        width = self.width.get() or 1
        c = self.c2 if (rot * COLOR_SEGMENTS >> 12) & 1 else self.c1
        self.c1.update()
        self.c2.update()
        cr = c.r
        cg = c.g
        cb = c.b
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
                frame[k] = cr * b // 255
                frame[k + 1] = cg * b // 255
                frame[k + 2] = cb * b // 255
            else:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
