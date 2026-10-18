"""A thin horizontal plane of light sweeping up and down the tree."""

from effects._util import ONE, Layout, Param, Phase

NAME = "plane_sweep"
DURATION_MS = 15000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "thickness": (0.03, 0.01, 0.2),
}
NEEDS_COORDS = True

PERIOD_MS = 5300  # one up-and-down sweep at speed 1
COLOR = (0, 255, 0)


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h
        self.speed = Param(ctx, "speed", 1.0)
        self.thickness = Param(ctx, "thickness", 0.03)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        ph = self.phase.update(t_ms, self.speed.get())
        pos = 2 * ph if ph < ONE // 2 else 2 * (ONE - ph)
        thick = self.thickness.get() or 1
        cr, cg, cb = COLOR
        known = self.known
        h = self.h
        for i in range(self.n):
            k = 3 * i
            d = h[i] - pos
            if d < 0:
                d = -d
            if known[i] and d < thick:
                b = 255 - d * 255 // thick
                frame[k] = cr * b >> 8
                frame[k + 1] = cg * b >> 8
                frame[k + 2] = cb * b >> 8
            else:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
