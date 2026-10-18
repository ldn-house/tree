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
BIN_SHIFT = 8  # distance bins of 256 stored units
BINS = 65536 >> BIN_SHIFT


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
        # Per ripple, filled in when it spawns: the LEDs ordered by squared
        # distance from its centre (a counting sort into BINS distance
        # bins), with those distances alongside, so each frame only visits
        # the LEDs in the distance band its ring currently covers.
        self.order = [array("H", bytes(2 * n)) for _ in range(SLOTS)]
        self.dist = [array("H", bytes(2 * n)) for _ in range(SLOTS)]
        self.bin_start = [array("H", bytes(2 * (BINS + 1))) for _ in range(SLOTS)]
        self.tmp = array("H", bytes(2 * n))
        self.cursor = array("H", bytes(2 * (BINS + 1)))
        self.start = array("i", bytes(4 * SLOTS))
        self.color = bytearray(SLOTS)
        self.alive = bytearray(SLOTS)
        # Each frame starts from this background (ambient on every lit
        # position) and rings add into it, clamping at 255. Every
        # contribution is positive, so that equals summing and then clamping.
        ar, ag, ab = AMBIENT
        self.bg = bytearray(3 * n)
        for i in range(n):
            if lay.known[i]:
                self.bg[3 * i] = ar
                self.bg[3 * i + 1] = ag
                self.bg[3 * i + 2] = ab
        self.frame = None
        self.view = None
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
        lit = self.lit
        tmp = self.tmp
        cursor = self.cursor
        start = self.bin_start[slot]
        order = self.order[slot]
        dist = self.dist[slot]
        for b in range(BINS + 1):
            cursor[b] = 0
        m = self.m
        for j in range(m):
            i = lit[j]
            dx = px[i] - cx
            dy = py[i] - cy
            dz = pz[i] - cz
            d = (dx * dx + dy * dy + dz * dz) >> D2_SHIFT
            if d > 65535:
                d = 65535
            tmp[j] = d
            cursor[d >> BIN_SHIFT] += 1
        total = 0
        for b in range(BINS):
            n_b = cursor[b]
            start[b] = total
            cursor[b] = total
            total += n_b
        start[BINS] = total
        for j in range(m):
            d = tmp[j]
            b = d >> BIN_SHIFT
            k = cursor[b]
            cursor[b] = k + 1
            order[k] = lit[j]
            dist[k] = d
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

        # Ambient everywhere first (a C-level copy).
        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(self.bg)] = self.bg
        rmax = self.radius.get()
        shell = self.shell
        last_bin = BINS - 1
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
            order = self.order[s]
            dist = self.dist[s]
            start = self.bin_start[s]
            # Only the distance bins the ring's shell overlaps.
            b0 = (lo2 + 1) >> BIN_SHIFT
            b1 = (hi2 - 1) >> BIN_SHIFT
            if b1 > last_bin:
                b1 = last_bin
            for k in range(start[b0], start[b1 + 1]):
                d = dist[k]
                if d > lo2 and d < hi2:
                    e = d - r2
                    if e < 0:
                        e = -e
                    e = e // span
                    if e < shell:
                        q = (shell - e) * 256 // shell
                        b = q * q >> 8
                        vr = cr * b >> 8
                        vg = cg * b >> 8
                        vb = cb * b >> 8
                        if vr or vg or vb:
                            j = 3 * order[k]
                            v = frame[j] + vr
                            frame[j] = v if v < 255 else 255
                            v = frame[j + 1] + vg
                            frame[j + 1] = v if v < 255 else 255
                            v = frame[j + 2] + vb
                            frame[j + 2] = v if v < 255 else 255

