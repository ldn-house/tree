"""F1 chequered flag: a chequered flag waving across the front of the tree.

The tree is cut into square columns as seen from the front (+z, the
calibration's front camera), alternately white and dark, so front and back
LEDs agree and the board reads as one flag from the room. A wave runs
across it from left to right, bending the rows up and down like cloth in
the wind, and the flag drifts slowly upwards. Crests catch the light and
troughs are a little dimmer, which reads as folds. Smooth motion only: no
LED switches faster than the board moves past it.

White is sized from the power budget for the half of the tree that's lit,
so the cap barely engages.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, SIN, Layout, Param, Phase

NAME = "f1_chequered"
DURATION_MS = 15000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "rows": (5, 3, 9),  # squares from the bottom to the top
}
NEEDS_COORDS = True
AUTO = False

WAVE_MS = 1600  # one ripple passes at speed 1
DRIFT_MS = 5000  # the flag drifts up one row in this long
AMP = 90  # ripple height, /256 of a row
WAVES = 1  # ripples across the tree's width
DARK = 6  # /256 of white for the dark squares, so the tree's shape stays visible
SHADE = 70  # /256: troughs this much dimmer than crests
SHARE = 0.95  # lit half's share of the budget


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h
        # Horizontal position in tree heights (Layout's x is in radii), offset
        # so it's positive.
        self.ux = array("H", bytes(2 * n))
        rmax = int(lay.rmax * 65536)
        for i in range(n):
            x = lay.x[i] * rmax
            self.ux[i] = ONE + (x // 65536 if x >= 0 else -(-x // 65536))
        count = lay.count or n
        # Per-channel white level for a lit LED, half the tree lit.
        w = int(SHARE * budget_for_current() / (count / 2) / 3)
        self.white = w if w < 255 else 255
        self.speed = Param(ctx, "speed", 1.0)
        self.rows = Param(ctx, "rows", 5, 1)
        self.wave = Phase(WAVE_MS)
        self.drift = Phase(DRIFT_MS)

    def render(self, t_ms, frame):
        speed = self.speed.get()
        wave = self.wave.update(t_ms, speed) >> 4  # 0..255
        drift = self.drift.update(t_ms, speed)  # 0..ONE-1: one row
        rows = self.rows.get()
        rows = 3 if rows < 3 else 9 if rows > 9 else rows
        row = ONE // rows
        amp = AMP * row >> 8
        white = self.white
        dark = white * DARK >> 8
        off = row - (drift * row >> 12)  # keeps y positive
        h = self.h
        ux = self.ux
        known = self.known
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            x = ux[i]
            # The ripple travels across: SIN is the cloth's height here.
            s = SIN[((x * WAVES >> 4) - wave) & 255]
            y = h[i] + ((s - 128) * amp >> 7) + off
            if (y // row + x // row) & 1:
                lv = white - (white * SHADE * (255 - s) >> 16)
            else:
                lv = dark
            frame[k] = lv
            frame[k + 1] = lv
            frame[k + 2] = lv
