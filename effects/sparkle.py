"""Random white sparkles: a fresh random set of LEDs lights every tick."""

NAME = "sparkle"
DURATION_MS = 8000
PARAMS = {
    "density": (0.03, 0.005, 0.3),
    "interval_ms": (110, 30, 1000),
}
NEEDS_COORDS = False


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.params = ctx.params
        self.rand = ctx.random.getrandbits
        self.lit = bytearray(ctx.n)
        self.tick = -1
        self.raw = None
        self.threshold = 0

    def render(self, t_ms, frame):
        p = self.params
        d = p.get("density", 0.03)
        if d is not self.raw:
            self.raw = d
            self.threshold = int(d * 65536)
        tick = t_ms // int(p.get("interval_ms", 110))
        lit = self.lit
        if tick != self.tick:
            self.tick = tick
            rand = self.rand
            th = self.threshold
            for i in range(self.n):
                lit[i] = 1 if rand(16) < th else 0
        for i in range(self.n):
            v = 255 if lit[i] else 0
            k = 3 * i
            frame[k] = v
            frame[k + 1] = v
            frame[k + 2] = v
