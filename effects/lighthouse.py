"""A warm lighthouse beam sweeping round the tree.

A vertical wedge through the trunk axis rotates about it. The leading edge
is crisp, and the trailing edge fades out more slowly like a lamp's
afterglow. The rest of the tree keeps a faint warm ambient glow.
"""

from effects._util import ONE, MASK, Layout, Param, Phase

NAME = "lighthouse"
DURATION_MS = 20000
PARAMS = {
    "speed": (1.0, 0.2, 4.0),
    "width": (50, 10, 120),  # beam width, degrees
}
NEEDS_COORDS = True

PERIOD_MS = 5000  # one revolution at speed 1
BEAM = (255, 196, 120)
AMBIENT = (6, 3, 1)
TRAIL = 2  # the trailing edge fades over TRAIL x the half-width


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.theta = lay.theta
        self.speed = Param(ctx, "speed", 1.0)
        self.width = Param(ctx, "width", 50, 1)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        # theta = atan2(z, x) increases anticlockwise seen from above, so
        # the beam turns that way.
        beam = self.phase.update(t_ms, self.speed.get())
        half = max(1, self.width.get() * ONE // 720)
        tail = half * TRAIL
        br, bg, bb = BEAM
        ar, ag, ab = AMBIENT
        mid = ONE >> 1
        known = self.known
        theta = self.theta
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            # Signed angle from the beam's centre: positive ahead of it.
            d = ((theta[i] - beam + mid) & MASK) - mid
            if d >= 0:
                w = half
            else:
                d = -d
                w = tail
            if d < w:
                # Quadratic falloff from the centre to the edge.
                q = (w - d) * 256 // w
                b = q * q >> 8
                frame[k] = ar + ((br - ar) * b >> 8)
                frame[k + 1] = ag + ((bg - ag) * b >> 8)
                frame[k + 2] = ab + ((bb - ab) * b >> 8)
            else:
                frame[k] = ar
                frame[k + 1] = ag
                frame[k + 2] = ab
