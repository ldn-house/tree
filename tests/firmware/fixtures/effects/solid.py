NAME = "solid"
DURATION_MS = 4000
# tint 1: warm white (255, 87, 14) scaled by level, for the colour floor.
PARAMS = {"level": (255, 0, 255), "tint": (0, 0, 1)}
NEEDS_COORDS = False

WARM = (255, 87, 14)


class Effect:
    def __init__(self, ctx):
        self.ctx = ctx

    def render(self, t_ms, frame):
        v = int(self.ctx.params.get("level", 255))
        if int(self.ctx.params.get("tint", 0)):
            r, g, b = WARM
            r = r * v // 255
            g = g * v // 255
            b = b * v // 255
            for k in range(0, len(frame), 3):
                frame[k] = r
                frame[k + 1] = g
                frame[k + 2] = b
            return
        for i in range(len(frame)):
            frame[i] = v
