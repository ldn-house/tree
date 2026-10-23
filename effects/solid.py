"""Solid: the whole tree in one colour.

This is what a colour picked on Home Assistant's light shows when the
selected effect has no colour of its own (firmware/app/ha.py). Not in the
auto cycle.
"""

from effects._util import ColorParam, rgb_spec

NAME = "solid"
DURATION_MS = 20000
PARAMS = {
    "color": rgb_spec(255, 147, 41),  # warm white
}
NEEDS_COORDS = False
AUTO = False


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.color = ColorParam(ctx, "color", PARAMS["color"][0])

    def render(self, t_ms, frame):
        c = self.color
        c.update()
        r = c.r
        g = c.g
        b = c.b
        for k in range(0, 3 * self.n, 3):
            frame[k] = r
            frame[k + 1] = g
            frame[k + 2] = b
