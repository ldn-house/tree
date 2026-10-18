"""Ripples of colour expanding through the tree like raindrops on a pond.

Each drop starts at a random LED's position and grows as a 3D spherical
shell that slows as it widens and fades as it grows. Several overlap at
once, in colours from a soft palette, over a faint blue ambient.
"""

from array import array

from effects._util import Layout, Param

NAME = "raindrops"
DURATION_MS = 20000
PARAMS = {
    "rate": (1.5, 0.2, 4.0),  # new drops per second
    "radius": (0.4, 0.1, 0.8),  # how far a ripple spreads, in tree heights
}
NEEDS_COORDS = True

SLOTS = 6  # ripples alive at once, at most
LIFE_MS = 1800  # each ripple's lifetime
SHELL = 0.05  # half-thickness of the ripple shell, tree heights
PALETTE = bytes((
    0, 190, 210,  # teal
    120, 70, 255,  # violet
    255, 70, 140,  # rose
    255, 150, 40,  # amber
    40, 130, 255,  # blue
    70, 255, 150,  # mint
))
AMBIENT = (0, 0, 1)
U = 1024  # positions in 1/U tree heights
D2_SHIFT = 5  # squared distances stored >> D2_SHIFT to fit 16 bits


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
        lay = Layout(ctx)
        self.known = lay.known
        # Isotropic positions in 1/U tree heights: Layout's x/z are in units
        # of the tree's radius (lay.rmax tree heights) x 4096.
        scale = int(lay.rmax * U)
        self.px = array("h", bytes(2 * n))
        self.py = array("h", bytes(2 * n))
        self.pz = array("h", bytes(2 * n))
        self.lit = array("H", bytes(2 * n))  # indices of LEDs with positions
        m = 0
        for i in range(n):
            if lay.known[i]:
                self.px[i] = lay.x[i] * scale >> 12
                self.py[i] = lay.h[i] * U >> 12
                self.pz[i] = lay.z[i] * scale >> 12
                self.lit[m] = i
                m += 1
        self.m = m
        # Per ripple: squared distance from its centre to every LED, filled
        # in when it spawns so each frame only compares numbers.
        self.d2 = [array("H", bytes(2 * n)) for _ in range(SLOTS)]
        self.start = array("i", bytes(4 * SLOTS))
        self.color = bytearray(SLOTS)
        self.alive = bytearray(SLOTS)
        self.acc = array("H", bytes(6 * n))
        self.rate = Param(ctx, "rate", 1.5, 65536 / 1000)  # spawn chance per ms, /65536
        self.radius = Param(ctx, "radius", 0.4, U)
        self.shell = int(SHELL * U)
        self.last = 0

    def spawn(self, slot, t_ms):
        if not self.m:
            return
        c = self.lit[self.rand(16) % self.m]
        cx = self.px[c]
        cy = self.py[c]
        cz = self.pz[c]
        px = self.px
        py = self.py
        pz = self.pz
        d2 = self.d2[slot]
        for i in range(self.n):
            dx = px[i] - cx
            dy = py[i] - cy
            dz = pz[i] - cz
            d = (dx * dx + dy * dy + dz * dz) >> D2_SHIFT
            d2[i] = d if d < 65535 else 65535
        self.start[slot] = t_ms
        self.color[slot] = self.rand(8) % (len(PALETTE) // 3)
        self.alive[slot] = 1

    def render(self, t_ms, frame):
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0
        alive = self.alive
        # Spawn at the requested rate, and always keep at least one ripple.
        free = -1
        busy = 0
        for s in range(SLOTS):
            if alive[s] and t_ms - self.start[s] >= LIFE_MS:
                alive[s] = 0
            if alive[s]:
                busy += 1
            elif free < 0:
                free = s
        if free >= 0 and (busy == 0 or self.rand(16) < self.rate.get() * dt):
            self.spawn(free, t_ms)

        n = self.n
        acc = self.acc
        for i in range(3 * n):
            acc[i] = 0
        rmax = self.radius.get()
        shell = self.shell
        known = self.known
        for s in range(SLOTS):
            if not alive[s]:
                continue
            # Ease out: fast at first, slowing as it spreads.
            a = (t_ms - self.start[s]) * 256 // LIFE_MS  # age 0..255
            inv = 256 - a
            r = rmax * (65536 - inv * inv) >> 16
            # Fade linearly with age.
            fade = inv  # 256..0
            lo = r - shell
            lo2 = (lo * lo >> D2_SHIFT) if lo > 0 else -1
            hi = r + shell
            hi2 = hi * hi >> D2_SHIFT
            # |d - r| ~ |d2 - r2| / 2r, in the stored units.
            r2 = r * r >> D2_SHIFT
            span = max(r, shell) * 2 >> D2_SHIFT or 1
            o = self.color[s] * 3
            cr = PALETTE[o] * fade >> 8
            cg = PALETTE[o + 1] * fade >> 8
            cb = PALETTE[o + 2] * fade >> 8
            d2 = self.d2[s]
            for i in range(n):
                d = d2[i]
                if d > lo2 and d < hi2 and known[i]:
                    e = d - r2
                    if e < 0:
                        e = -e
                    e = e // span
                    if e < shell:
                        q = (shell - e) * 256 // shell
                        b = q * q >> 8
                        k = 3 * i
                        acc[k] += cr * b >> 8
                        acc[k + 1] += cg * b >> 8
                        acc[k + 2] += cb * b >> 8
        ar, ag, ab = AMBIENT
        for i in range(n):
            k = 3 * i
            if known[i]:
                v = acc[k] + ar
                frame[k] = v if v < 255 else 255
                v = acc[k + 1] + ag
                frame[k + 1] = v if v < 255 else 255
                v = acc[k + 2] + ab
                frame[k + 2] = v if v < 255 else 255
            else:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
