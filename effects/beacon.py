"""Beacon: a sonar ping from the star, or a slow amber hazard double-flash.

Style 0 (sonar): every couple of seconds a ring of light leaves the top of
the tree and travels down the cone as a growing spherical shell centred on
the star, widening and fading as it goes, over a dim blue ambient glow. Its
leading (lower) edge is crisp and its trailing edge fades slowly.

Style 1 (hazard): the whole tree gives two soft amber flashes, then rests,
like a warning beacon, brightest at the top. About one flash a second on
average, never more than three a second.
"""

from array import array

from effects._util import ONE, Layout, Param, Phase

NAME = "beacon"
DURATION_MS = 20000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "style": (0, 0, 1),  # 0 = sonar ping, 1 = amber hazard double-flash
}
NEEDS_COORDS = True

# Sonar.
PERIOD_MS = 2600  # one sonar ping per period at speed 1
TRAVEL_MS = 2000  # of each period, the ring is travelling for this long
PING = (70, 210, 255)
AMBIENT = (0, 3, 9)
FRONT0 = 0.03  # leading half-width at the star, tree heights
FRONT1 = 0.07  # added by the time it reaches the bottom
TRAIL = 3  # the trailing edge fades over TRAIL x the leading half-width
FADE = 150  # /256 of the ring's brightness lost on the way down, linearly
TAIL = 128  # over the last TAIL/256 of the trip it also fades out to nothing

# Hazard.
HAZARD_MS = 2200  # one double-flash per period at speed 1
FLASH2_MS = 420  # second flash starts this long after the first
RISE_MS = 70
HOLD_MS = 110
FALL_MS = 170
AMBER = (255, 105, 0)
AMBER_REST = (5, 1, 0)
LOW = 110  # /256 brightness at the bottom of the tree; the top is 256


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        # Distance of each LED from the star (top of the trunk axis), in
        # ONE = one tree height. Layout's x/z are in units of the outer
        # radius, which is lay.rmax tree heights.
        rm = lay.rmax
        self.dist = array("H", bytes(2 * n))
        dmax = 1
        for i in range(n):
            if lay.known[i]:
                dx = lay.x[i] * rm
                dz = lay.z[i] * rm
                dy = ONE - lay.h[i]
                d = int((dx * dx + dy * dy + dz * dz) ** 0.5)
                self.dist[i] = d
                if d > dmax:
                    dmax = d
        self.front0 = int(FRONT0 * ONE)
        self.front1 = int(FRONT1 * ONE)
        # Travel past the lowest LED by the final trailing width, so the
        # ring's tail has left the tree by the end of the trip.
        self.dmax = dmax + (self.front0 + self.front1) * TRAIL
        # Hazard brightness by height: LOW at the bottom up to 256 at the top.
        self.grad = array("H", bytes(2 * n))
        for i in range(n):
            self.grad[i] = LOW + (256 - LOW) * lay.h[i] // ONE
        self.speed = Param(ctx, "speed", 1.0)
        self.style = Param(ctx, "style", 0, 1)
        self.ping = Phase(PERIOD_MS)
        self.hazard = Phase(HAZARD_MS)

    def render(self, t_ms, frame):
        speed = self.speed.get()
        ph = self.ping.update(t_ms, speed)
        hz = self.hazard.update(t_ms, speed)
        if self.style.get():
            self.render_hazard(hz, frame)
        else:
            self.render_sonar(ph, frame)

    def render_sonar(self, ph, frame):
        ar, ag, ab = AMBIENT
        pr, pg, pb = PING
        travel = TRAVEL_MS * ONE // PERIOD_MS
        if ph < travel:
            a = ph * 256 // travel  # 0..255 along the trip
            # Half linear, half ease-out: quick past the sparse top, slower
            # over the wide, dense bottom.
            ease = (a + (a * (512 - a) >> 8)) >> 1
            front = self.dmax * ease >> 8
            wf = self.front0 + (self.front1 * a >> 8)
            amp = 256 - (FADE * a >> 8)
            if a > 256 - TAIL:
                # Ease out to zero, so the ring dissolves instead of
                # cutting to the ambient glow.
                q = (256 - a) * 256 // TAIL
                amp = amp * (q * q >> 8) >> 8
        else:
            amp = 0
            front = 0
            wf = 1
        wt = wf * TRAIL
        known = self.known
        dist = self.dist
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            e = dist[i] - front
            if e >= 0:
                w = wf
            else:
                e = -e
                w = wt
            if amp and e < w:
                q = (w - e) * 256 // w
                b = (q * q >> 8) * amp >> 8
                frame[k] = ar + ((pr - ar) * b >> 8)
                frame[k + 1] = ag + ((pg - ag) * b >> 8)
                frame[k + 2] = ab + ((pb - ab) * b >> 8)
            else:
                frame[k] = ar
                frame[k + 1] = ag
                frame[k + 2] = ab

    def render_hazard(self, hz, frame):
        t = hz * HAZARD_MS >> 12  # ms into the period at speed 1
        lv = flash(t)
        v = flash(t - FLASH2_MS)
        if v > lv:
            lv = v
        cr, cg, cb = AMBER
        rr, rg, rb = AMBER_REST
        known = self.known
        grad = self.grad
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            b = lv * grad[i] >> 8
            frame[k] = rr + ((cr - rr) * b >> 8)
            frame[k + 1] = rg + ((cg - rg) * b >> 8)
            frame[k + 2] = rb + ((cb - rb) * b >> 8)


def flash(d):
    """Envelope (0..256) of one flash d ms after it starts."""
    if d < 0:
        return 0
    if d < RISE_MS:
        return 256 * d // RISE_MS
    if d < RISE_MS + HOLD_MS:
        return 256
    if d < RISE_MS + HOLD_MS + FALL_MS:
        v = 256 * (RISE_MS + HOLD_MS + FALL_MS - d) // FALL_MS
        return v * v >> 8
    return 0
