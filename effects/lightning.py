"""A thunderstorm: lightning bolts striking down through a dim storm cloud.

The tree glows a dim, slowly rolling blue-grey, brightest at the top where
the cloud is. Every few seconds, at random, a bolt forks down through the
tree along a jagged path of neighbouring LEDs: a nearest-neighbour graph
built from the coordinates, walked downwards at random, sometimes with one
branch. A strike is a dim leader crawling down the path, a bright return
stroke, sometimes one re-flash, then an afterglow fading to violet.

Photosensitivity: only the bolt's own LEDs (and a faint halo next to them)
light up; the rest of the tree stays at the cloud ambient. A strike has at
most two flashes, a third of a second apart, and strikes are at least
MIN_GAP_MS apart whatever the storm setting.
"""

from array import array

from effects._util import ONE, SIN, Layout, Param

NAME = "lightning"
DURATION_MS = 30000
PARAMS = {
    "storm": (1.0, 0.3, 3.0),  # how often bolts strike
}
NEEDS_COORDS = True

MEAN_GAP_MS = 6000  # mean time between strikes at storm 1
MIN_GAP_MS = 2500  # strikes never start closer together than this
MIN_EXTRA_MS = 600  # random spread kept above the minimum gap in a wild storm
FIRST_MS = 900  # the first strike comes soon after the effect starts
K = 5  # downward neighbours kept per LED
REACH = 0.12  # longest step of a bolt, tree heights
MAX_PATH = 72  # LEDs in a bolt, main path plus branch
BRANCH_CHANCE = 160  # out of 256
REFLASH_CHANCE = 128  # out of 256: a strike has one re-flash
LEADER_MS = 200  # the leader takes this long to reach the bottom
STROKE_MS = 70  # return stroke at full brightness
REFLASH_AT = 330  # re-flash starts this long after the return stroke
REFLASH_MS = 50
GLOW_MS = 800  # afterglow fades over this

BOLT = (210, 215, 255)  # return stroke
LEADER = (90, 100, 170)
GLOW = (110, 70, 255)  # afterglow tint
CLOUD = (6, 7, 12)  # ambient at the top of the tree
CLOUD_BASE = 0.45  # lower edge of the cloud, tree heights; dark below
BRANCH = 150  # branch brightness relative to the main channel, /256
HALO = 64  # neighbours of the channel, /256
U = 1024  # positions in 1/U tree heights


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
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

        # For each LED, its K nearest neighbours that are lower down, within
        # REACH, nearest first; 0xFFFF pads. LEDs sorted by height bound the
        # scan to the band just below each LED.
        order = sorted((i for i in range(n) if lay.known[i]), key=lambda i: py[i])
        reach = int(REACH * U)
        reach2 = reach * reach
        down = array("H", b"\xff\xff" * (K * n))
        dist = array("i", bytes(4 * K))  # squared distances of the kept ones
        for a in range(len(order)):
            i = order[a]
            xi = px[i]
            yi = py[i]
            zi = pz[i]
            row = K * i
            kept = 0
            b = a - 1
            while b >= 0 and yi - py[order[b]] <= reach:
                j = order[b]
                b -= 1
                dy = yi - py[j]
                if dy <= 0:
                    continue
                dx = px[j] - xi
                if dx > reach or dx < -reach:
                    continue
                dz = pz[j] - zi
                d2 = dx * dx + dy * dy + dz * dz
                if d2 > reach2 or (kept == K and d2 >= dist[K - 1]):
                    continue
                # Insert in order of distance, dropping the farthest.
                m = kept if kept < K else K - 1
                while m > 0 and dist[m - 1] > d2:
                    dist[m] = dist[m - 1]
                    down[row + m] = down[row + m - 1]
                    m -= 1
                dist[m] = d2
                down[row + m] = j
                if kept < K:
                    kept += 1
        self.down = down
        # Bolts start among the highest LEDs.
        top = [i for i in order if py[i] >= U * 85 // 100] or order[-8:]
        self.tops = array("H", top)

        # Ambient: CLOUD at the top, fading down to CLOUD_BASE; below that the
        # tree is dark (a fade to zero would pass through levels too dim for
        # the channels to keep their hue). Rolled by a slow wave whose phase
        # depends on angle and height.
        self.level = bytearray(n)
        self.wave = bytearray(n)
        base = int(CLOUD_BASE * ONE)
        for i in range(n):
            q = (lay.h[i] - base) * 256 // (ONE - base)
            # At least 90, so even the cloud's dimmest LEDs (with the wave at
            # its low) keep every channel lit and look grey-blue.
            self.level[i] = 0 if q <= 0 else min(255, 90 + (165 * q * q >> 16))
            self.wave[i] = (lay.theta[i] * 256 // ONE + lay.h[i] * 384 // ONE) & 255

        self.path = array("H", bytes(2 * MAX_PATH))
        self.depth = array("H", bytes(2 * MAX_PATH))  # step along the bolt
        self.main = bytearray(MAX_PATH)  # 1 on the main channel
        self.count = 0
        self.steps = 1
        self.reflash = 0
        self.start = -1000000  # current strike's start
        self.next = FIRST_MS + self.rand(9)
        self.storm = Param(ctx, "storm", 1.0, 256)
        self.storm.get()  # convert the float now, not mid-render

    def walk(self, i, d, room, main):
        # Random walk down the graph from LED i (already on the path at
        # depth d), adding at most room LEDs. Returns the depth reached.
        down = self.down
        rand = self.rand
        c = self.count
        while room > 0:
            m = 0
            while m < K and down[K * i + m] != 0xFFFF:
                m += 1
            if m == 0:
                break
            # Mostly one of the nearest three, sometimes a longer jump.
            k = rand(8) % (3 if m > 3 and rand(2) else m)
            i = down[K * i + k]
            d += 1
            self.path[c] = i
            self.depth[c] = d
            self.main[c] = main
            c += 1
            room -= 1
        self.count = c
        return d

    def strike(self, t_ms):
        rand = self.rand
        tops = self.tops
        self.count = 0
        if not len(tops):
            return
        i = tops[rand(16) % len(tops)]
        self.path[0] = i
        self.depth[0] = 0
        self.main[0] = 1
        self.count = 1
        steps = self.walk(i, 0, MAX_PATH * 2 // 3, 1)
        main_len = self.count
        if main_len > 6 and rand(8) < BRANCH_CHANCE:
            # Fork from somewhere in the upper half of the main channel.
            f = 2 + rand(8) * (main_len // 2 - 2) // 256
            self.walk(self.path[f], self.depth[f], min(MAX_PATH - main_len, main_len // 2), 0)
        self.steps = steps if steps > 0 else 1
        self.reflash = 1 if rand(8) < REFLASH_CHANCE else 0
        self.start = t_ms

    def schedule(self, t_ms):
        storm = self.storm.get()
        if storm < 16:
            storm = 16
        # MIN_GAP_MS plus a random extra averaging the rest of the mean gap
        # (at least MIN_EXTRA_MS, so a wild storm still strikes irregularly).
        extra = MEAN_GAP_MS * 256 // storm - MIN_GAP_MS
        if extra < MIN_EXTRA_MS:
            extra = MIN_EXTRA_MS
        self.next = t_ms + MIN_GAP_MS + (extra * self.rand(9) >> 8)

    def render(self, t_ms, frame):
        if t_ms >= self.next:
            self.strike(t_ms)
            self.schedule(t_ms)
        age = t_ms - self.start

        # Ambient cloud.
        cr, cg, cb = CLOUD
        roll = t_ms // 40
        known = self.known
        level = self.level
        wave = self.wave
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            v = level[i] * (150 + (SIN[(wave[i] + roll) & 255] * 106 >> 8)) >> 8
            frame[k] = cr * v >> 8
            frame[k + 1] = cg * v >> 8
            frame[k + 2] = cb * v >> 8

        count = self.count
        if count == 0 or age < 0:
            return
        # Channel brightness (0-256) and colour for this moment of the strike.
        lead = LEADER_MS
        glow_from = lead + STROKE_MS
        if self.reflash:
            glow_from = lead + REFLASH_AT + REFLASH_MS
        if age >= glow_from + GLOW_MS:
            self.count = 0
            return
        br, bg, bb = BOLT
        gr, gg, gb = GLOW
        tip = -1
        if age < lead:
            # Leader: the tip crawls down, leaving a faint trail.
            tip = age * self.steps // lead
            r, g, b = LEADER
            s = 0
        elif age < lead + STROKE_MS:
            r, g, b = br, bg, bb
            s = 256
        elif self.reflash and age < lead + REFLASH_AT:
            # Between the return stroke and the re-flash: dims quickly.
            q = age - lead - STROKE_MS
            span = REFLASH_AT - STROKE_MS
            s = 40 + 140 * (span - q) * (span - q) // (span * span)
            r, g, b = br, bg, bb
        elif self.reflash and age < glow_from:
            r, g, b = br, bg, bb
            s = 210
        else:
            # Afterglow: quadratic fade, shifting from white to violet.
            q = age - glow_from
            w = (GLOW_MS - q) * 256 // GLOW_MS
            s = 180 * w * w >> 16
            r = gr + ((br - gr) * w >> 8)
            g = gg + ((bg - gg) * w >> 8)
            b = gb + ((bb - gb) * w >> 8)

        path = self.path
        depth = self.depth
        main = self.main
        down = self.down
        for c in range(count):
            i = path[c]
            if tip >= 0:
                d = depth[c]
                if d > tip:
                    continue
                # Brightest at the tip, a dim trail above it.
                v = 256 if d == tip else 70
            else:
                v = s
            if not main[c]:
                v = v * BRANCH >> 8
            k = 3 * i
            x = r * v >> 8
            if frame[k] < x:
                frame[k] = x
            x = g * v >> 8
            if frame[k + 1] < x:
                frame[k + 1] = x
            x = b * v >> 8
            if frame[k + 2] < x:
                frame[k + 2] = x
            if tip >= 0:
                continue
            # A faint halo on the nearest neighbours below.
            v = v * HALO >> 8
            hr = r * v >> 8
            hg = g * v >> 8
            hb = b * v >> 8
            for m in range(2):
                j = down[K * i + m]
                if j == 0xFFFF:
                    break
                k = 3 * j
                if frame[k] < hr:
                    frame[k] = hr
                if frame[k + 1] < hg:
                    frame[k + 1] = hg
                if frame[k + 2] < hb:
                    frame[k + 2] = hb
