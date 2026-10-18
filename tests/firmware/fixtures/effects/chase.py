NAME = "chase"
DURATION_MS = 4000
PARAMS = {"speed": (1.0, 0.1, 5.0)}
NEEDS_COORDS = False


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.ctx = ctx

    def render(self, t_ms, frame):
        on = int(t_ms * self.ctx.params.get("speed", 1.0) / 50) % 3
        for i in range(self.n):
            v = 255 if i % 3 == on else 0
            frame[3 * i] = v
            frame[3 * i + 1] = 0
            frame[3 * i + 2] = 0
