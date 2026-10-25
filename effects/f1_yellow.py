"""F1 yellow flag: a yellow flag waved from side to side.

The tree's left and right halves (seen from the front) take turns, with a
soft boundary between them, about once a second each: every LED rises and
falls smoothly between a low glow and full yellow. Nothing strobes and the
tree is never dark, so it stays well under photosensitivity limits (no
more than one cycle a second at speed 1, three at the maximum speed).

Levels are sized from the power budget, so the swing is visible instead of
being flattened by the 1.5 A cap: the two halves together stay just under it.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, SIN, ColorParam, Layout, Param, Phase, rgb_spec

NAME = "f1_yellow"
DURATION_MS = 12000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "color": rgb_spec(255, 255, 0),
}
NEEDS_COORDS = True
AUTO = False

PERIOD_MS = 1200  # one wave (left, then right) at speed 1
HIGH_SHARE = 1.5  # a lit half: per-LED channel sum at 1.5x the tree's average budget
LOW_SHARE = 0.35  # the resting half
EDGE = 0.25  # boundary softness, in tree radii either side of the middle


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        count = lay.count or n
        per_led = budget_for_current() / count
        self.high = int(HIGH_SHARE * per_led)
        self.low = int(LOW_SHARE * per_led)
        # side[i]: 0 = fully left, 256 = fully right.
        edge = int(EDGE * ONE) or 1
        self.side = array("H", bytes(2 * n))
        for i in range(n):
            if lay.known[i]:
                v = 128 + lay.x[i] * 128 // edge
                self.side[i] = 0 if v < 0 else 256 if v > 256 else v
            else:
                self.side[i] = 128
        self.speed = Param(ctx, "speed", 1.0)
        self.phase = Phase(PERIOD_MS)
        self.color = ColorParam(ctx, "color", PARAMS["color"][0])
        self.scale = 256  # colour -> per-LED sum at `high`

    def render(self, t_ms, frame):
        c = self.color
        if c.update():
            s = c.r + c.g + c.b
            self.scale = self.high * 256 // s if s else 0
        ph = self.phase.update(t_ms, self.speed.get())
        w = SIN[ph >> 4]  # 0..255: how far the right half is lit
        low = self.low * 256 // self.high if self.high else 0  # /256 of full
        span = 256 - low
        lr = low + (span * w >> 8)
        ll = low + (span * (255 - w) >> 8)
        scale = self.scale
        cr = c.r * scale >> 8
        cg = c.g * scale >> 8
        cb = c.b * scale >> 8
        if cr > 255:
            cr = 255
        if cg > 255:
            cg = 255
        if cb > 255:
            cb = 255
        side = self.side
        known = self.known
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            sd = side[i]
            lv = (lr * sd + ll * (256 - sd)) >> 8
            frame[k] = cr * lv >> 8
            frame[k + 1] = cg * lv >> 8
            frame[k + 2] = cb * lv >> 8
