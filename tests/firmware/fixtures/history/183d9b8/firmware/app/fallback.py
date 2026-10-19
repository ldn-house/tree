"""Built-in effect used when no packaged effect can run, so a broken or
missing effects package still leaves the tree lit (and power-capped)."""

NAME = "fallback"
DURATION_MS = 60000
PARAMS = {}
NEEDS_COORDS = False


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n

    def render(self, t_ms, frame):
        for k in range(0, 3 * self.n, 3):
            frame[k] = 255
            frame[k + 1] = 140
            frame[k + 2] = 50
