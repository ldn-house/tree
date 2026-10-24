"""Solid warm white glow.

The tone param picks one of several warm whites, for comparing on the real
LEDs (an A/B for the colour, from Home Assistant's "tone" number). The
default is 2, which looked warm on the tree, like incandescent Christmas
lights, once low levels stopped being dithered:
- 0: (255, 140, 50), the original colour picked on a screen (looks cool and
  yellow-white on the LEDs);
- 1: 0 white-balanced by (1.0, 0.85, 0.7), less green and blue;
- 2: 0 through gamma 1.8, which is what a screen shows for 0 in linear
  light (the LEDs' PWM is linear), the warmest.
A tone 3, 0 through gamma 2.2 (255, 68, 7), was dropped: on the tree it
looked orange rather than white. A saved tone 3 shows tone 2.
For a solid colour, baking gamma into the colour gives the same hue as a
global GAMMA setting would, at any brightness.
"""

NAME = "warm_white"
DURATION_MS = 10000
PARAMS = {
    "tone": (2, 0, 2),
}
NEEDS_COORDS = False

TONES = (
    (255, 140, 50),
    (255, 119, 35),
    (255, 87, 14),
)


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.params = ctx.params

    def render(self, t_ms, frame):
        tone = int(self.params.get("tone", 2))
        if tone < 0 or tone >= len(TONES):
            tone = 2
        r, g, b = TONES[tone]
        for k in range(0, 3 * self.n, 3):
            frame[k] = r
            frame[k + 1] = g
            frame[k + 2] = b
