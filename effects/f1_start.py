"""F1 start: the green flag climbs the tree, then holds.

Formula 1 race starts and green flags (Home Assistant's F1 takeover). From
a dark tree, green bands land one after another from the bottom up, the
whole tree flashes once and settles into a steady green glow that it then
keeps: unlike level_up, of which this is a one-level variant, it never
starts another level. At speed 3 the climb takes about 1.6 s.
"""

from effects import level_up
from effects._util import ColorParam, rgb_spec

NAME = "f1_start"
DURATION_MS = 12000
PARAMS = {
    "speed": (3.0, 0.5, 6.0),  # band rate; 1.0 = level_up's beat
    "steps": (8, 3, 16),  # height bands
    "finish": (0, 0, 1),  # 0 = one flash settling to the glow, 1 = one slow pulse
    "color": rgb_spec(0, 255, 0),
}
NEEDS_COORDS = True
AUTO = False


class Effect(level_up.Effect):
    once = True

    def __init__(self, ctx):
        level_up.Effect.__init__(self, ctx)
        self.palette = bytearray(3)
        self.color = ColorParam(ctx, "color", PARAMS["color"][0])

    def render(self, t_ms, frame):
        c = self.color
        if c.update():
            p = self.palette
            p[0] = c.r
            p[1] = c.g
            p[2] = c.b
        level_up.Effect.render(self, t_ms, frame)
