NAME = "height"
DURATION_MS = 4000
PARAMS = {}
NEEDS_COORDS = True


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.h = bytearray(ctx.n)
        for i, c in enumerate(ctx.coords):
            self.h[i] = 0 if c is None else int(max(0.0, min(1.0, c[1])) * 255)

    def render(self, t_ms, frame):
        for i in range(self.n):
            frame[3 * i] = 0
            frame[3 * i + 1] = self.h[i]
            frame[3 * i + 2] = 255 - self.h[i]
