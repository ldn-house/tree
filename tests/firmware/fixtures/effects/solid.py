NAME = "solid"
DURATION_MS = 4000
PARAMS = {"level": (255, 0, 255)}
NEEDS_COORDS = False


class Effect:
    def __init__(self, ctx):
        self.ctx = ctx

    def render(self, t_ms, frame):
        v = int(self.ctx.params.get("level", 255))
        for i in range(len(frame)):
            frame[i] = v
