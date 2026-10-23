NAME = "tint"
DURATION_MS = 4000
# A colour param (docs/effect-api.md): a packed 0xRRGGBB int.
PARAMS = {"color": (0x102030, 0, 0xFFFFFF, "rgb"), "level": (255, 0, 255)}
NEEDS_COORDS = False
AUTO = False  # selectable and usable by takeovers, never auto-cycled


class Effect:
    def __init__(self, ctx):
        self.ctx = ctx

    def render(self, t_ms, frame):
        c = self.ctx.params.get("color", 0x102030)
        lv = int(self.ctx.params.get("level", 255))
        r = (c >> 16) * lv // 255
        g = (c >> 8 & 255) * lv // 255
        b = (c & 255) * lv // 255
        for k in range(0, len(frame), 3):
            frame[k] = r
            frame[k + 1] = g
            frame[k + 2] = b
