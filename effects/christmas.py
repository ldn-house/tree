"""Red and green sectors rotating around the trunk.

LEDs without a position fall back to a red/green theater chase.
"""

from effects._util import ONE, MASK, Layout, Param, Phase

NAME = "christmas"
DURATION_MS = 30000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "sectors": (6, 2, 12),
}
NEEDS_COORDS = False

PERIOD_MS = 4800  # one full rotation at speed 1
CHASE_STEPS = 60  # fallback chase advances 60 steps per rotation


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.theta = lay.theta
        self.speed = Param(ctx, "speed", 1.0)
        self.sectors = Param(ctx, "sectors", 6, 1)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        rot = self.phase.update(t_ms, self.speed.get())
        sectors = self.sectors.get() & ~1 or 2
        step = rot * CHASE_STEPS >> 12
        known = self.known
        theta = self.theta
        for i in range(self.n):
            k = 3 * i
            if known[i]:
                red = not (((theta[i] + rot) & MASK) * sectors >> 12) & 1
                frame[k] = 255 if red else 0
                frame[k + 1] = 0 if red else 255
            elif i % 3 == step % 3:
                red = not (i // 3) & 1
                frame[k] = 255 if red else 0
                frame[k + 1] = 0 if red else 255
            else:
                frame[k] = 0
                frame[k + 1] = 0
            frame[k + 2] = 0
