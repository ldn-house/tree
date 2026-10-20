"""Beacon: a sonar ping from the star, or a slow amber hazard double-flash.

Style 0 (sonar): rings of light leave the star one after another and
travel down the cone as growing spherical shells centred on it, widening
and fading as they go, so two or three are always in flight. Each ring is
half a step further through a palette than the one before, and a dim
ambient glow drifts through the same colours. Leading (lower) edges are
crisp and trailing edges fade slowly. Ring spacing, palette and loop line
up, so it repeats every LOOP_MS with no visible restart.

Style 1 (hazard): the whole tree gives two soft amber flashes, then rests,
like a warning beacon, brightest at the top. About one flash a second on
average, never more than three a second.
"""

from array import array

from effects._util import ONE, Layout, Param, Phase

NAME = "beacon"
DURATION_MS = 20400  # two sonar loops (LOOP_MS below)
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "style": (0, 0, 1),  # 0 = sonar ping, 1 = amber hazard double-flash
}
NEEDS_COORDS = True

# Sonar.
SPACING_MS = 850  # a new ring leaves the star this often, at speed 1
LIFE_MS = 2550  # each ring travels for this long, so three are in flight
PALETTE = bytes((
    40, 170, 255,  # azure
    0, 230, 160,  # teal
    120, 255, 40,  # lime
    255, 170, 0,  # amber
    255, 40, 120,  # rose
    140, 60, 255,  # violet
))
STEPS = 2  # rings per palette colour (each ring is half a step on)
RINGS = STEPS * len(PALETTE) // 3  # rings per loop
LOOP_MS = RINGS * SPACING_MS  # 10.2 s: rings, palette and loop line up
PERIOD_MS = LOOP_MS  # the web export's exact loop
AMBIENT = 7  # /256 of the drifting palette colour as the ambient glow
RING = 230  # /256 ring peak, so three rings stay under the power cap
FRONT0 = 0.03  # leading half-width at the star, tree heights
FRONT1 = 0.07  # added by the time it reaches the bottom
TRAIL = 3  # the trailing edge fades over TRAIL x the leading half-width
FADE = 150  # /256 of the ring's brightness lost on the way down, linearly
TAIL = 128  # over the last TAIL/256 of the trip it also fades out to nothing
DBIN = 64  # distance bins (1/4096 tree heights) for the sorted LED order

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
        # Known LEDs sorted by distance from the star, in DBIN-wide bins,
        # so each ring only visits the LEDs near its shell.
        nb = dmax // DBIN + 2
        self.nbins = nb
        self.bin_start = array("H", bytes(2 * (nb + 1)))
        self.order = array("H", bytes(2 * n))
        counts = [0] * nb
        for i in range(n):
            if lay.known[i]:
                counts[self.dist[i] // DBIN] += 1
        total = 0
        for b in range(nb):
            self.bin_start[b] = total
            total += counts[b]
            counts[b] = self.bin_start[b]
        self.bin_start[nb] = total
        for i in range(n):
            if lay.known[i]:
                b = self.dist[i] // DBIN
                self.order[counts[b]] = i
                counts[b] += 1
        counts = None
        # Hazard brightness by height: LOW at the bottom up to 256 at the top.
        self.grad = array("H", bytes(2 * n))
        for i in range(n):
            self.grad[i] = LOW + (256 - LOW) * lay.h[i] // ONE
        self.bg = bytearray(3 * n)
        self.amb = -1
        self.frame = None
        self.view = None
        self.speed = Param(ctx, "speed", 1.0)
        self.style = Param(ctx, "style", 0, 1)
        self.ping = Phase(PERIOD_MS)
        self.hazard = Phase(HAZARD_MS)

    def render(self, t_ms, frame):
        speed = self.speed.get()
        self.ping.update(t_ms, speed)
        hz = self.hazard.update(t_ms, speed)
        if self.style.get():
            self.render_hazard(hz, frame)
        else:
            # ms into the loop at speed 1, exact (the phase accumulates
            # ms x ONE modulo LOOP_MS x ONE).
            self.render_sonar(self.ping.acc >> 12, frame)

    def render_sonar(self, t, frame):
        # Ambient: the palette colour at this point of the loop, drifting
        # continuously (one colour per STEPS rings).
        ncol = len(PALETTE) // 3
        pos = t * 256 // (SPACING_MS * STEPS)  # palette position, /256
        c = (pos >> 8) % ncol
        f = pos & 255
        o = 3 * c
        o2 = 3 * ((c + 1) % ncol)
        ar = (PALETTE[o] * (256 - f) + PALETTE[o2] * f) * AMBIENT >> 16
        ag = (PALETTE[o + 1] * (256 - f) + PALETTE[o2 + 1] * f) * AMBIENT >> 16
        ab = (PALETTE[o + 2] * (256 - f) + PALETTE[o2 + 2] * f) * AMBIENT >> 16
        # The ambient changes a step every few frames: rebuild the
        # background only then, and copy it in one go.
        amb = (ar << 16) | (ag << 8) | ab
        bg = self.bg
        if amb != self.amb:
            self.amb = amb
            known = self.known
            for i in range(self.n):
                if known[i]:
                    k = 3 * i
                    bg[k] = ar
                    bg[k + 1] = ag
                    bg[k + 2] = ab
        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(bg)] = bg
        # Rings: the newest left at k_now * SPACING_MS; older ones are
        # still in flight for LIFE_MS. Ring indices wrap with the loop.
        k_now = t // SPACING_MS
        for j in range(LIFE_MS // SPACING_MS + 1):
            k = k_now - j
            age = t - k * SPACING_MS
            if age >= LIFE_MS:
                continue
            # Ring colour: half a palette step per ring.
            r = k % RINGS
            o = 3 * (r // STEPS)
            if r % STEPS:
                o2 = 3 * ((r // STEPS + 1) % ncol)
                pr = (PALETTE[o] + PALETTE[o2]) >> 1
                pg = (PALETTE[o + 1] + PALETTE[o2 + 1]) >> 1
                pb = (PALETTE[o + 2] + PALETTE[o2 + 2]) >> 1
            else:
                pr = PALETTE[o]
                pg = PALETTE[o + 1]
                pb = PALETTE[o + 2]
            self.ring(age * 256 // LIFE_MS, pr, pg, pb, frame)

    def ring(self, a, pr, pg, pb, frame):
        """Add one ring, a/256 of the way along its trip, in colour p."""
        # Half linear, half ease-out: quick past the sparse top, slower
        # over the wide, dense bottom.
        ease = (a + (a * (512 - a) >> 8)) >> 1
        front = self.dmax * ease >> 8
        wf = self.front0 + (self.front1 * a >> 8)
        wt = wf * TRAIL
        amp = (256 - (FADE * a >> 8)) * RING >> 8
        if a > 256 - TAIL:
            # Ease out to zero, so the ring dissolves into the ambient.
            q = (256 - a) * 256 // TAIL
            amp = amp * (q * q >> 8) >> 8
        if not amp:
            return
        b0 = (front - wt) // DBIN
        b1 = (front + wf) // DBIN
        if b0 < 0:
            b0 = 0
        if b1 > self.nbins - 1:
            b1 = self.nbins - 1
        if b0 > b1:
            return
        dist = self.dist
        order = self.order
        for j in range(self.bin_start[b0], self.bin_start[b1 + 1]):
            i = order[j]
            e = dist[i] - front
            if e >= 0:
                w = wf
            else:
                e = -e
                w = wt
            if e < w:
                q = (w - e) * 256 // w
                b = (q * q >> 8) * amp >> 8
                k = 3 * i
                v = frame[k] + (pr * b >> 8)
                frame[k] = v if v < 255 else 255
                v = frame[k + 1] + (pg * b >> 8)
                frame[k + 1] = v if v < 255 else 255
                v = frame[k + 2] + (pb * b >> 8)
                frame[k + 2] = v if v < 255 else 255

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
