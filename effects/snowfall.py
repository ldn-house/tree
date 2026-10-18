"""Snow drifting down through the tree.

Flakes fall through the cone's volume, swaying gently. Their horizontal
position is kept as a point in a unit disc (taken from a random LED, so
flakes fall where the lights are), scaled by the tree's measured radius at
the flake's height, so they spread out as the tree widens. LEDs
near a flake glow softly. Some flakes settle on a lower branch and linger
there briefly before melting away, and a new flake starts at the top.
"""

import math
from array import array

from effects._util import SIN, Layout, Param

NAME = "snowfall"
DURATION_MS = 30000
PARAMS = {
    "density": (1.0, 0.2, 3.0),
    "speed": (1.0, 0.3, 3.0),
}
NEEDS_COORDS = True

BASE_FLAKES = 30  # flakes at density 1
MAX_FLAKES = 90  # at density 3
FALL = 160  # fall speed at speed 1, 1/U tree heights per second (+-30%)
GLOW = 0.05  # radius of a flake's glow, tree heights
SWAY = 0.025  # sideways sway amplitude, tree heights
SETTLE_CHANCE = 77  # out of 256: about 30% of flakes settle on a branch
LINGER_MS = 2200  # how long a settled flake stays before melting
FLAKE = (190, 215, 255)
SETTLED = (255, 255, 255)
AMBIENT = (0, 0, 2)
U = 1024  # positions in 1/U tree heights
BANDS = 16  # height bands for the radius profile and the LED index
SUB = 16  # flake heights are kept in 1/(U*SUB) for smooth slow motion


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
        self.params = ctx.params
        lay = Layout(ctx)
        self.known = lay.known
        scale = int(lay.rmax * U)
        px = array("h", bytes(2 * n))
        py = array("h", bytes(2 * n))
        pz = array("h", bytes(2 * n))
        for i in range(n):
            if lay.known[i]:
                px[i] = lay.x[i] * scale >> 12
                py[i] = lay.h[i] * U >> 12
                pz[i] = lay.z[i] * scale >> 12
        self.px = px
        self.py = py
        self.pz = pz

        # LEDs sorted by height, with the start of each height band, so a
        # flake only checks LEDs near its own height.
        bw = U // BANDS
        order = sorted((i for i in range(n) if lay.known[i]), key=lambda i: py[i])
        self.order = array("H", order)
        self.band_start = array("H", bytes(2 * (BANDS + 2)))
        b = 0
        for k, i in enumerate(order):
            while b <= BANDS and py[i] >= b * bw:
                self.band_start[b] = k
                b += 1
        while b <= BANDS + 1:
            self.band_start[b] = len(order)
            b += 1
        self.bw = bw

        # The tree's radius per height band (90th percentile of the LEDs'
        # horizontal distance), smoothed so it never grows with height.
        radius = array("h", bytes(2 * (BANDS + 1)))
        for b in range(BANDS):
            lo = self.band_start[b]
            hi = self.band_start[b + 1]
            rs = sorted(px[order[k]] * px[order[k]] + pz[order[k]] * pz[order[k]] for k in range(lo, hi))
            radius[b] = int(math.sqrt(rs[int(0.9 * (len(rs) - 1))])) if rs else 0
        for b in range(BANDS - 1, -1, -1):
            radius[b] = max(radius[b], radius[b + 1], U // 40)
        radius[BANDS] = radius[BANDS - 1]
        self.radius = radius

        m = MAX_FLAKES
        self.fy = array("i", bytes(4 * m))  # height, 1/(U*SUB)
        self.fu = array("h", bytes(2 * m))  # position in the unit disc, +-256
        self.fv = array("h", bytes(2 * m))
        self.vel = array("H", bytes(2 * m))  # fall speed, 1/U per second
        self.sway = bytearray(m)  # sway phase
        self.target = array("h", bytes(2 * m))  # settle below this height, or -1
        self.landed = array("h", bytes(2 * m))  # LED it settled on, or -1
        self.since = array("i", bytes(4 * m))  # when it settled
        # Start with flakes already spread through the tree; none of these
        # settle, so the first frames don't show a burst of settled flakes.
        for f in range(m):
            self.spawn(f, self.rand(10) * (U + U // 16) // 1024, 0)
            self.target[f] = -1
        self.acc = array("H", bytes(2 * n))
        self.glow = int(GLOW * U)
        self.sway_amp = int(SWAY * U)
        self.speed = Param(ctx, "speed", 1.0, 256)
        self.density = Param(ctx, "density", 1.0, BASE_FLAKES)
        self.last = 0

    def spawn(self, f, y, t_ms):
        rand = self.rand
        self.fy[f] = y * SUB
        # Take the disc position of a random LED (its horizontal position
        # over the tree's radius at its height), jittered, so flakes fall
        # through where the lights are rather than through empty middle.
        u = rand(6) - 32
        v = rand(6) - 32
        m = len(self.order)
        if m:
            i = self.order[rand(16) % m]
            r = self.radius[self.py[i] // self.bw] or 1
            u += self.px[i] * 256 // r
            v += self.pz[i] * 256 // r
        self.fu[f] = max(-256, min(256, u))
        self.fv[f] = max(-256, min(256, v))
        self.vel[f] = FALL * (179 + (rand(7) * 154 >> 7)) >> 8  # 70%-130%
        self.sway[f] = rand(8)
        # Settlers pick a branch height in the lower 60% of the tree.
        self.target[f] = rand(9) * (U * 6 // 10) >> 9 if rand(8) < SETTLE_CHANCE else -1
        self.landed[f] = -1
        self.since[f] = t_ms

    def render(self, t_ms, frame):
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0
        count = self.density.get()
        if count > MAX_FLAKES:
            count = MAX_FLAKES
        if count < 1:
            count = 1
        speed = self.speed.get()
        n = self.n
        acc = self.acc
        for i in range(n):
            acc[i] = 0
        px = self.px
        py = self.py
        pz = self.pz
        order = self.order
        band_start = self.band_start
        bw = self.bw
        radius = self.radius
        glow = self.glow
        glow2 = glow * glow
        top = (U + U // 16) * SUB
        for f in range(count):
            led = self.landed[f]
            if led >= 0:
                # Settled: hold, then melt.
                age = t_ms - self.since[f]
                if age >= LINGER_MS:
                    self.spawn(f, U + U // 16, t_ms)
                    continue
                b = 255 - age * 255 // LINGER_MS
                v = acc[led] + 1024 + b  # values over 1024 mark settled LEDs
                acc[led] = v if v < 2047 else 2047
                continue
            y = self.fy[f] - (self.vel[f] * dt * speed >> 8) * SUB // 1000
            if y < -glow * SUB:
                self.spawn(f, U + U // 16, t_ms)
                continue
            self.fy[f] = y
            y = y // SUB
            # Horizontal position: disc point x tree radius at this height,
            # plus a gentle sway.
            b = y // bw
            r = radius[b] if 0 <= b <= BANDS else (radius[0] if b < 0 else radius[BANDS])
            s = self.sway[f] + (t_ms >> 5)
            sx = (SIN[s & 255] - 128) * self.sway_amp >> 7
            sz = (SIN[(s + 64) & 255] - 128) * self.sway_amp >> 7
            x = (self.fu[f] * r >> 8) + sx
            z = (self.fv[f] * r >> 8) + sz
            lo = (y - glow) // bw
            hi = (y + glow) // bw + 1
            if lo < 0:
                lo = 0
            if hi > BANDS + 1:
                hi = BANDS + 1
            if lo >= hi:
                continue
            best = -1
            best_d2 = glow2
            for k in range(band_start[lo], band_start[hi]):
                i = order[k]
                dx = px[i] - x
                if dx >= glow or dx <= -glow:
                    continue
                dz = pz[i] - z
                if dz >= glow or dz <= -glow:
                    continue
                dy = py[i] - y
                d2 = dx * dx + dy * dy + dz * dz
                if d2 < glow2:
                    q = (glow2 - d2) * 255 // glow2
                    acc[i] += q * q >> 8
                    if d2 < best_d2:
                        best_d2 = d2
                        best = i
            if best >= 0 and self.target[f] >= 0 and y <= self.target[f]:
                self.landed[f] = best
                self.since[f] = t_ms

        fr, fg, fb = FLAKE
        sr, sg, sb = SETTLED
        ar, ag, ab = AMBIENT
        known = self.known
        for i in range(n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            v = acc[i]
            if v >= 1024:
                # A settled flake (plus any falling glow on top).
                v -= 1024
                if v > 255:
                    v = 255
                frame[k] = ar + ((sr - ar) * v >> 8)
                frame[k + 1] = ag + ((sg - ag) * v >> 8)
                frame[k + 2] = ab + ((sb - ab) * v >> 8)
            else:
                if v > 255:
                    v = 255
                frame[k] = ar + ((fr - ar) * v >> 8)
                frame[k + 1] = ag + ((fg - ag) * v >> 8)
                frame[k + 2] = ab + ((fb - ab) * v >> 8)
