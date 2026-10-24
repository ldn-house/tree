"""F1 red flag: a slow red pulse rolling down the tree.

The whole tree breathes red, a little later towards the bottom, so the
pulse seems to roll down from the star. One breath every 2.4 s at speed 1
between a quarter and full brightness: slow and smooth, never dark, never
a flash.

Levels are sized from the power budget, so the cap doesn't flatten the
pulse: the peak just reaches it.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, SIN, ColorParam, Layout, Param, Phase, rgb_spec

NAME = "f1_red"
DURATION_MS = 12000
PARAMS = {
    "speed": (1.0, 0.25, 2.0),
    "color": rgb_spec(255, 0, 0),
}
NEEDS_COORDS = False
AUTO = False

PERIOD_MS = 2400
LOW = 64  # /256 of the peak at the bottom of a breath
LAG = 1024  # the bottom of the tree lags the top by this much of a period (/ONE)


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        count = lay.count or n
        self.peak_sum = int(budget_for_current() / count)
        # Phase offset per LED: 0 at the top, LAG at the bottom (by string
        # order where the position is unknown, which Layout gives as h).
        self.lag = array("H", bytes(2 * n))
        for i in range(n):
            self.lag[i] = (ONE - lay.h[i]) * LAG // ONE
        self.speed = Param(ctx, "speed", 1.0)
        self.phase = Phase(PERIOD_MS)
        self.color = ColorParam(ctx, "color", PARAMS["color"][0])
        self.lut = bytearray(3 * 256)  # colour at each breath level

    def render(self, t_ms, frame):
        c = self.color
        if c.update():
            s = c.r + c.g + c.b
            scale = self.peak_sum * 256 // s if s else 0
            if scale > 256:
                scale = 256
            lut = self.lut
            for v in range(256):
                lv = (LOW + ((256 - LOW) * v >> 8)) * scale >> 8
                lut[3 * v] = c.r * lv >> 8
                lut[3 * v + 1] = c.g * lv >> 8
                lut[3 * v + 2] = c.b * lv >> 8
        ph = self.phase.update(t_ms, self.speed.get())
        lut = self.lut
        lag = self.lag
        for i in range(self.n):
            # Breath shape: sine starting at its trough, squared for longer rests.
            s = SIN[((ph - lag[i]) >> 4) & 255]
            v = s * s >> 8
            j = 3 * v
            k = 3 * i
            frame[k] = lut[j]
            frame[k + 1] = lut[j + 1]
            frame[k + 2] = lut[j + 2]
