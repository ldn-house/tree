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
BANDS = 16  # height bands for the radius profile
CELL = 64  # grid cell size for finding LEDs near a flake, 1/U tree heights
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
        # A grid over (height, x) in CELL-sized cells, LEDs sorted by cell,
        # so a flake scans one short run of LEDs per grid row near it.
        xs = [px[i] for i in order]
        self.xmin = min(xs) if xs else 0
        cols = ((max(xs) if xs else 0) - self.xmin) // CELL + 1
        rows = U // CELL + 2
        cells = sorted(order, key=lambda i: (py[i] // CELL) * cols + (px[i] - self.xmin) // CELL)
        self.grid = array("H", cells)
        self.cell_start = array("H", bytes(2 * (rows * cols + 1)))
        c = 0
        for k, i in enumerate(cells):
            cell = (py[i] // CELL) * cols + (px[i] - self.xmin) // CELL
            while c <= cell:
                self.cell_start[c] = k
                c += 1
        while c <= rows * cols:
            self.cell_start[c] = len(cells)
            c += 1
        self.rows = rows
        self.cols = cols

        # Each frame starts from this background (ambient on every lit
        # position) and only the LEDs near flakes are repainted.
        ar, ag, ab = AMBIENT
        self.bg = bytearray(3 * n)
        for i in range(n):
            if lay.known[i]:
                self.bg[3 * i] = ar
                self.bg[3 * i + 1] = ag
                self.bg[3 * i + 2] = ab
        self.frame = None
        self.view = None
        self.acc = array("H", bytes(2 * n))
        self.touched = array("H", bytes(2 * n))
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
        acc = self.acc
        touched = self.touched
        nt = 0
        px = self.px
        py = self.py
        pz = self.pz
        grid = self.grid
        cell_start = self.cell_start
        cols = self.cols
        last_row = self.rows - 1
        xmin = self.xmin
        bw = self.bw
        radius = self.radius
        glow = self.glow
        glow2 = glow * glow
        fy = self.fy
        fu = self.fu
        fv = self.fv
        vel = self.vel
        sway = self.sway
        target = self.target
        landed = self.landed
        since = self.since
        amp = self.sway_amp
        fall = dt * speed
        tsway = t_ms >> 5
        for f in range(count):
            led = landed[f]
            if led >= 0:
                # Settled: hold, then melt.
                age = t_ms - since[f]
                if age >= LINGER_MS:
                    self.spawn(f, U + U // 16, t_ms)
                    continue
                v = acc[led]
                if v == 0:
                    touched[nt] = led
                    nt += 1
                v += 1024 + 255 - age * 255 // LINGER_MS  # over 1024 marks settled
                acc[led] = v if v < 2047 else 2047
                continue
            y = fy[f] - (vel[f] * fall >> 8) * SUB // 1000
            if y < -glow * SUB:
                self.spawn(f, U + U // 16, t_ms)
                continue
            fy[f] = y
            y = y // SUB
            # Horizontal position: disc point x tree radius at this height,
            # plus a gentle sway.
            b = y // bw
            r = radius[b] if 0 <= b <= BANDS else (radius[0] if b < 0 else radius[BANDS])
            s = sway[f] + tsway
            x = (fu[f] * r >> 8) + ((SIN[s & 255] - 128) * amp >> 7)
            z = (fv[f] * r >> 8) + ((SIN[(s + 64) & 255] - 128) * amp >> 7)
            r0 = (y - glow) // CELL
            r1 = (y + glow) // CELL
            if r0 < 0:
                r0 = 0
            if r1 > last_row:
                r1 = last_row
            c0 = (x - glow - xmin) // CELL
            c1 = (x + glow - xmin) // CELL
            if c0 < 0:
                c0 = 0
            if c1 >= cols:
                c1 = cols - 1
            if r0 > r1 or c0 > c1:
                continue
            best = -1
            best_d2 = glow2
            for row in range(r0, r1 + 1):
                base = row * cols
                for k in range(cell_start[base + c0], cell_start[base + c1 + 1]):
                    i = grid[k]
                    dz = pz[i] - z
                    if dz >= glow or dz <= -glow:
                        continue
                    dx = px[i] - x
                    dy = py[i] - y
                    d2 = dx * dx + dy * dy + dz * dz
                    if d2 < glow2:
                        q = (glow2 - d2) * 255 // glow2
                        q = q * q >> 8
                        if q:
                            v = acc[i]
                            if v == 0:
                                touched[nt] = i
                                nt += 1
                            acc[i] = v + q
                        if d2 < best_d2:
                            best_d2 = d2
                            best = i
            if best >= 0 and 0 <= target[f] and y <= target[f]:
                landed[f] = best
                since[f] = t_ms

        # Ambient everywhere (a C-level copy), then paint the LEDs touched.
        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(self.bg)] = self.bg
        fr, fg, fb = FLAKE
        sr, sg, sb = SETTLED
        ar, ag, ab = AMBIENT
        for j in range(nt):
            i = touched[j]
            v = acc[i]
            acc[i] = 0
            if v == 0:
                continue
            k = 3 * i
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
