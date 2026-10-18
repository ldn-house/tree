"""Rainbow gradient by height that scrolls through the colour wheel.

LEDs without a position fall back to a gradient along the string.
"""

from effects._util import WHEEL, Layout, Param, Phase

NAME = "rainbow_cycle"
DURATION_MS = 20000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
}
NEEDS_COORDS = False

PERIOD_MS = 4700  # one trip round the colour wheel at speed 1


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.hue = Layout(ctx).h
        self.speed = Param(ctx, "speed", 1.0)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        ph = self.phase.update(t_ms, self.speed.get()) >> 4
        hue = self.hue
        for i in range(self.n):
            o = (((hue[i] >> 4) + ph) & 255) * 3
            k = 3 * i
            frame[k] = WHEEL[o]
            frame[k + 1] = WHEEL[o + 1]
            frame[k + 2] = WHEEL[o + 2]
