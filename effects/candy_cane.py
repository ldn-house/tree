"""Red and white horizontal stripes drifting along the tree.

LEDs without a position fall back to stripes along the string.
"""

from effects._util import ONE, MASK, Layout, Param, Phase

NAME = "candy_cane"
DURATION_MS = 10000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "stripes": (8, 2, 20),
}
NEEDS_COORDS = False

PERIOD_MS = 10000  # stripes move one stripe pair per period at speed 1
FALLBACK_WIDTH = 10  # LEDs per stripe along the string
FALLBACK_STEPS = 100  # string offset advances 100 LEDs per period


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h
        self.speed = Param(ctx, "speed", 1.0)
        self.stripes = Param(ctx, "stripes", 8, 1)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        ph = self.phase.update(t_ms, self.speed.get())
        stripes = self.stripes.get()
        offset = ph * FALLBACK_STEPS >> 12
        half = ONE >> 1
        known = self.known
        h = self.h
        for i in range(self.n):
            if known[i]:
                red = ((h[i] * stripes + ph) & MASK) < half
            else:
                red = not ((i + offset) // FALLBACK_WIDTH) & 1
            k = 3 * i
            frame[k] = 255
            frame[k + 1] = 0 if red else 255
            frame[k + 2] = 0 if red else 255
