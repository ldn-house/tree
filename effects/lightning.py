"""Lightning: bolts striking down through a dark tree.

Between strikes the tree is dark. Every few seconds, at random, a bolt
forks down through the tree along a jagged path of neighbouring LEDs (see
Bolt below): a dim leader, a bright return stroke, sometimes one re-flash,
then an afterglow fading to violet.

Photosensitivity: only the bolt's own LEDs (and a faint halo next to them)
light up. A strike has at most two flashes, a third of a second apart, and
strikes are at least MIN_GAP_MS apart whatever the frequency.
"""

from array import array

from effects._util import Layout, Param

NAME = "lightning"
DURATION_MS = 30000
PARAMS = {
    "frequency": (1.0, 0.3, 3.0),  # how often bolts strike
}
NEEDS_COORDS = True


class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
        self.bolt = Bolt(ctx, Layout(ctx))
        self.dark = bytearray(3 * ctx.n)
        self.frame = None
        self.view = None
        self.frequency = Param(ctx, "frequency", 1.0, 256)
        self.frequency.get()  # convert the float now, not mid-render

    def render(self, t_ms, frame):
        bolt = self.bolt
        bolt.update(t_ms, self.frequency.get())
        # Dark everywhere (a C-level copy), then the bolt.
        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(self.dark)] = self.dark
        bolt.paint(frame)


# --- Bolt (kept identical in lightning.py and storm.py) -------------------
# Hot-pushed effects can't bring a shared helper module with them (only
# effect modules can be pushed; see docs/effect-api.md), so each effect that
# draws bolts carries its own copy of this section.
#
# A bolt follows a jagged path of neighbouring LEDs: a nearest-neighbour
# graph built from the coordinates, walked downwards at random from one of
# the highest LEDs, sometimes with one branch. A strike is a dim leader
# crawling down the path, a bright return stroke, sometimes one re-flash,
# then an afterglow fading to violet. Only the bolt's own LEDs (and a faint
# halo next to them) light up; a strike has at most two flashes, a third of
# a second apart, and strikes are at least MIN_GAP_MS apart whatever the
# rate. Each frame: update(t_ms, rate), paint anything underneath, then
# paint(frame). After update(), flash (0-256) is the flash envelope (0
# outside the return stroke and re-flash) and origin is the bolt's first LED.

MEAN_GAP_MS = 6000  # mean time between strikes at rate 1
MIN_GAP_MS = 2500  # strikes never start closer together than this
MIN_EXTRA_MS = 600  # random spread kept above the minimum gap at high rates
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
BRANCH = 150  # branch brightness relative to the main channel, /256
HALO = 64  # neighbours of the channel, /256
U = 1024  # positions in 1/U tree heights


class Bolt:
    def __init__(self, ctx, lay, top=0.85):
        # top: bolts start among the LEDs above this height (tree heights).
        n = ctx.n
        self.rand = ctx.random.getrandbits
        scale = int(lay.rmax * U)
        # Positions in 1/U tree heights, shared with the effects using this.
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

        # Each LED's K nearest neighbours that are lower down, within REACH,
        # nearest first (0xFFFF pads), are found the first time a bolt
        # reaches it (see neighbours()) and kept. For that search, LEDs are
        # bucketed into columns REACH wide along x and sorted by height
        # within each, so a search scans the band just below an LED in
        # three columns.
        reach = int(REACH * U)
        self.reach = reach
        known = [i for i in range(n) if lay.known[i]]
        xmin = min(px[i] for i in known) if known else 0
        cols = ((max(px[i] for i in known) - xmin) // reach + 1) if known else 1
        cells = sorted(known, key=lambda i: ((px[i] - xmin) // reach) * U * 2 + py[i])
        start = array("H", bytes(2 * (cols + 1)))
        c = 0
        for k in range(len(cells)):
            col = (px[cells[k]] - xmin) // reach
            while c <= col:
                start[c] = k
                c += 1
        while c <= cols:
            start[c] = len(cells)
            c += 1
        self.cells = array("H", cells)
        self.cstart = start
        self.cols = cols
        self.xmin = xmin
        self.sx = array("h", (px[i] for i in cells))
        self.sy = array("h", (py[i] for i in cells))
        self.sz = array("h", (pz[i] for i in cells))
        self.down = array("H", b"\xff\xff" * (K * n))
        self.found = bytearray(n)
        self.dist = array("i", bytes(4 * K))
        t = int(top * U)
        tops = [i for i in known if py[i] >= t]
        if not tops and known:
            tops = sorted(known, key=lambda i: py[i])[-8:]
        self.tops = array("H", tops)

        self.path = array("H", bytes(2 * MAX_PATH))
        self.depth = array("H", bytes(2 * MAX_PATH))  # step along the bolt
        self.main = bytearray(MAX_PATH)  # 1 on the main channel
        self.count = 0
        self.steps = 1
        self.reflash = 0
        self.start = -1000000  # current strike's start
        self.next = FIRST_MS + self.rand(9)
        # Set by update() for paint() and the caller.
        self.flash = 0
        self.origin = 0
        self.tip = -1
        self.level = 0
        self.r = 0
        self.g = 0
        self.b = 0

    def neighbours(self, i):
        # Fills down[K*i ...] with LED i's nearest lower neighbours, once.
        if self.found[i]:
            return
        self.found[i] = 1
        px = self.px
        py = self.py
        pz = self.pz
        sx = self.sx
        sy = self.sy
        sz = self.sz
        cells = self.cells
        start = self.cstart
        down = self.down
        dist = self.dist
        reach = self.reach
        reach2 = reach * reach
        xi = px[i]
        yi = py[i]
        zi = pz[i]
        row = K * i
        kept = 0
        col = (xi - self.xmin) // reach
        c0 = col - 1 if col > 0 else 0
        c1 = col + 2 if col + 2 < self.cols else self.cols
        for cc in range(c0, c1):
            # First LED in this column at least REACH below i (binary
            # search), then on up to i's height.
            lo = start[cc]
            hi = start[cc + 1]
            floor = yi - reach
            while lo < hi:
                mid = (lo + hi) >> 1
                if sy[mid] < floor:
                    lo = mid + 1
                else:
                    hi = mid
            end = start[cc + 1]
            while lo < end:
                dy = yi - sy[lo]
                if dy <= 0:
                    break
                dx = sx[lo] - xi
                dz = sz[lo] - zi
                j = cells[lo]
                lo += 1
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

    def walk(self, i, d, room, main):
        # Random walk down the graph from LED i (already on the path at
        # depth d), adding at most room LEDs. Returns the depth reached.
        down = self.down
        rand = self.rand
        c = self.count
        while room > 0:
            self.neighbours(i)
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
        self.neighbours(i)  # for the halo in paint()
        self.count = c
        return d

    def strike(self, t_ms):
        rand = self.rand
        tops = self.tops
        self.count = 0
        if not len(tops):
            return
        i = tops[rand(16) % len(tops)]
        self.origin = i
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

    def schedule(self, t_ms, rate):
        # rate: strikes relative to MEAN_GAP_MS, fixed point 256 = 1.
        if rate < 16:
            rate = 16
        # MIN_GAP_MS plus a random extra averaging the rest of the mean gap
        # (at least MIN_EXTRA_MS, so a high rate still strikes irregularly).
        extra = MEAN_GAP_MS * 256 // rate - MIN_GAP_MS
        if extra < MIN_EXTRA_MS:
            extra = MIN_EXTRA_MS
        self.next = t_ms + MIN_GAP_MS + (extra * self.rand(9) >> 8)

    def update(self, t_ms, rate):
        if t_ms >= self.next:
            self.strike(t_ms)
            self.schedule(t_ms, rate)
        self.flash = 0
        self.tip = -1
        self.level = 0
        age = t_ms - self.start
        if self.count == 0 or age < 0:
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
        if age < lead:
            # Leader: the tip crawls down, leaving a faint trail.
            self.tip = age * self.steps // lead
            self.r, self.g, self.b = LEADER
        elif age < lead + STROKE_MS:
            self.r, self.g, self.b = BOLT
            self.level = 256
        elif self.reflash and age < lead + REFLASH_AT:
            # Between the return stroke and the re-flash: dims quickly.
            q = age - lead - STROKE_MS
            span = REFLASH_AT - STROKE_MS
            self.level = 40 + 140 * (span - q) * (span - q) // (span * span)
            self.r, self.g, self.b = BOLT
        elif self.reflash and age < glow_from:
            self.r, self.g, self.b = BOLT
            self.level = 210
        else:
            # Afterglow: quadratic fade, shifting from white to violet.
            q = age - glow_from
            w = (GLOW_MS - q) * 256 // GLOW_MS
            self.level = 180 * w * w >> 16
            gr, gg, gb = GLOW
            self.r = gr + ((br - gr) * w >> 8)
            self.g = gg + ((bg - gg) * w >> 8)
            self.b = gb + ((bb - gb) * w >> 8)
            return
        if self.tip < 0:
            self.flash = self.level

    def paint(self, frame):
        # Max-blends the bolt over whatever the frame holds.
        count = self.count
        if count == 0:
            return
        tip = self.tip
        s = self.level
        r = self.r
        g = self.g
        b = self.b
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
