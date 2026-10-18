"""Solid warm white glow."""

NAME = "warm_white"
DURATION_MS = 10000
PARAMS = {}
NEEDS_COORDS = False

WARM = (255, 140, 50)


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n

    def render(self, t_ms, frame):
        r, g, b = WARM
        for k in range(0, 3 * self.n, 3):
            frame[k] = r
            frame[k + 1] = g
            frame[k + 2] = b
