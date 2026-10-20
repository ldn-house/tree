"""Pong on the tree, in two styles.

Style 0 (wrapped): the cone's surface unrolled, with the angle round the
trunk as x (which wraps, so there are no side walls) and height as y. The
ball moves diagonally, spiralling round the tree as it climbs and falls,
between a paddle in a band at the bottom and one in a band near the top.
It moves in true arc length (the angle scaled by the cone's local
circumference), so it keeps the same size and speed all the way up.

Style 1 (volume): the ball bounces around inside the cone in 3D, off the
floor, a ceiling below the star and the sloping wall. The parts of the
wall facing left and right (+x and -x, the sides you see from the front)
are the goals, each with a paddle: a patch of the surface that slides
across its face and up and down.

In both, the ball is a glowing disc or sphere with a short comet trail,
where it strikes the paddle sets the angle it leaves at, and a miss
flashes the scorer's half of the tree once before the next serve. The
paddles are auto-played: a predictor runs a copy of the ball forward, a
slice per frame, to find where it will next reach a goal, and that paddle
heads there with a seeded error at a limited speed, so rallies end now
and then.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, Layout, Param

NAME = "pong"
DURATION_MS = 30000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "skill": (0.7, 0.0, 1.0),  # 1 = paddles aim perfectly, 0 = often wide
    "style": (0, 0, 1),  # 0 = wrapped round the tree, 1 = inside its volume
}
NEEDS_COORDS = True

# Positions in P = 1/65536 tree heights (style 0's x: 65536 per turn),
# velocities in P per ms; geometry works at 1/4096 to keep products small.
SERVE_V = 32  # 0.5 tree heights/s
MAX_V = 58
SPEEDUP = 18  # /256 faster after each return
MIN_ALONG = 140  # /256 of the speed kept along the goal axis
PADDLE_V = 30  # style 1 paddle speed, P/ms
PADDLE_A = 26  # style 0 paddle speed, 1/65536 turn per ms (0.4 turn/s)
STEP_MS = 16  # physics substep
PREDICT_STEPS = 25  # predictor substeps per frame

BALL_R = 0.11  # style 1 glow radius, tree heights
BALL_R2D = 0.08  # style 0
REACH = 0.035  # return reach beyond the paddle's lit edge, tree heights
# Style 1 (volume).
Y_LO = 0.07
Y_HI = 0.72
WALL = 0.82  # bounce at this share of the cone's local radius
SLOPE = 0.35  # the cone's radius shrinks ~this much per tree height
PAD_W = 0.11  # paddle half-width across the face
PAD_H = 0.1  # paddle half-height
# Style 0 (wrapped): paddle bands and where the ball turns.
BAND_LO = 0.12  # bottom paddle: LEDs below this height
BAND_HI = (0.7, 0.84)  # top paddle: LEDs in this band
TURN_LO = 0.15  # the ball's centre turns at these heights
TURN_HI = 0.68
PAD_A = 0.11  # paddle half-width, turns

BALL = (255, 235, 190)
SIDES = bytes((
    255, 90, 0,  # side 0 (+x / bottom): orange
    0, 150, 255,  # side 1 (-x / top): cyan
))
AMBIENT = (0, 0, 1)
HIT_MS = 260  # a paddle brightens briefly on a return
MISS_MS = 900  # the scorer's half flashes once
SERVE_MS = 700  # pause after a miss before the next serve
FLASH_SHARE = 0.9  # miss flash: the half tree at this share of the budget

YBINS = 32
BINS = 32  # radius profile bins

_PLAY = 0
_MISS = 1

# Ball state indices (style 0 uses _X for its angle).
_X = 0
_Y = 1
_Z = 2
_VX = 3
_VY = 4
_VZ = 5


def isqrt(n):
    if n <= 0:
        return 0
    x = n
    y = (x + 1) >> 1
    while y < x:
        x = y
        y = (x + n // x) >> 1
    return x


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
        lay = Layout(ctx)
        known = lay.known
        rm = lay.rmax
        # Per-LED positions in 1/1024 tree heights, and angle (1/65536 turn).
        self.px = array("h", bytes(2 * n))
        self.py = array("h", bytes(2 * n))
        self.pz = array("h", bytes(2 * n))
        self.pa = array("H", bytes(2 * n))
        for i in range(n):
            if known[i]:
                self.px[i] = int(lay.x[i] * rm / 4)
                self.py[i] = lay.h[i] >> 2
                self.pz[i] = int(lay.z[i] * rm / 4)
                self.pa[i] = lay.theta[i] << 4
        # Local cone radius (1/4096 tree heights): 85th percentile of the
        # radii per height bin, smoothed, non-increasing above the widest.
        radii = [[] for _ in range(BINS)]
        for i in range(n):
            if known[i]:
                radii[min(BINS - 1, lay.h[i] * BINS >> 12)].append(int(lay.r[i] * rm))
        sm = [0] * BINS
        last = int(0.3 * ONE)
        for b in range(BINS):
            rs = radii[b]
            if rs:
                rs.sort()
                last = rs[int(0.85 * (len(rs) - 1))]
            sm[b] = last
        radii = None
        sm = [sum(sm[max(0, b - 2):b + 3]) // len(sm[max(0, b - 2):b + 3]) for b in range(BINS)]
        widest = sm.index(max(sm))
        for b in range(widest + 1, BINS):
            sm[b] = min(sm[b], sm[b - 1])
        self.wall = array("H", [max(int(0.04 * ONE), int(r * WALL)) for r in sm])
        # Circumference per bin (1/4096 tree heights per turn), for style 0.
        self.circ = array("H", [max(200, int(6.2832 * r)) for r in sm])
        # LED lists, [style][side]. Style 1: goal faces (outer LEDs facing
        # +x / -x) and left/right halves. Style 0: the bottom and top
        # paddle bands, and bottom/top halves.
        lists = [[[], []], [[], []]]
        halves = [[[], []], [[], []]]
        for i in range(n):
            if not known[i]:
                continue
            b = min(BINS - 1, lay.h[i] * BINS >> 12)
            x = self.px[i]
            z = self.pz[i]
            if lay.r[i] * rm >= 0.6 * sm[b] and abs(x) >= 0.55 * abs(z):
                lists[1][0 if x > 0 else 1].append(i)
            halves[1][0 if x > 0 else 1].append(i)
            h = lay.h[i] / ONE
            if h < BAND_LO:
                lists[0][0].append(i)
            elif BAND_HI[0] <= h < BAND_HI[1]:
                lists[0][1].append(i)
            halves[0][0 if h < 0.4 else 1].append(i)
        self.pads = [[array("H", l) for l in s] for s in lists]
        self.halves = [[array("H", l) for l in s] for s in halves]
        # Miss flash level (/256) so a half lands at FLASH_SHARE of the budget.
        budget = budget_for_current()
        self.flash_lv = bytearray(4)
        for st in range(2):
            for s in range(2):
                c = SIDES[3 * s] + SIDES[3 * s + 1] + SIDES[3 * s + 2]
                cnt = len(halves[st][s]) or 1
                self.flash_lv[2 * st + s] = min(255, int(FLASH_SHARE * budget * 256 / (cnt * c)))
        lists = halves = None
        # LEDs ordered by height bin, so the ball only tests nearby bins.
        self.ybin_start = array("H", bytes(2 * (YBINS + 1)))
        self.yorder = array("H", bytes(2 * n))
        counts = [0] * YBINS
        for i in range(n):
            if known[i]:
                counts[min(YBINS - 1, self.py[i] * YBINS >> 10)] += 1
        total = 0
        for b in range(YBINS):
            self.ybin_start[b] = total
            total += counts[b]
            counts[b] = self.ybin_start[b]
        self.ybin_start[YBINS] = total
        for i in range(n):
            if known[i]:
                b = min(YBINS - 1, self.py[i] * YBINS >> 10)
                self.yorder[counts[b]] = i
                counts[b] += 1
        counts = None
        # Background (ambient on known LEDs), copied in one go per frame.
        self.bg = bytearray(3 * n)
        for i in range(n):
            if known[i]:
                self.bg[3 * i] = AMBIENT[0]
                self.bg[3 * i + 1] = AMBIENT[1]
                self.bg[3 * i + 2] = AMBIENT[2]
        self.frame = None
        self.view = None
        # Trail: brightness per LED, and the LEDs it covers (compacted as
        # they fade, so only those are decayed and drawn).
        self.trail = bytearray(n)
        self.act = array("H", bytes(2 * n))
        self.n_act = 0
        self.ball = array("i", bytes(4 * 6))
        self.pred = array("i", bytes(4 * 6))
        # Paddles: across (z, or style 0's angle) and height, their targets
        # and last return times. Index = side.
        self.pad_z = array("i", bytes(8))
        self.pad_y = array("i", bytes(8))
        self.tgt_z = array("i", bytes(8))
        self.tgt_y = array("i", bytes(8))
        self.hit_t = array("i", [-10000, -10000])
        self.pad_w = int(PAD_W * 65536)
        self.pad_h = int(PAD_H * 65536)
        self.pad_a = int(PAD_A * 65536)
        self.reach = int(REACH * 65536)
        self.y_lo = int(Y_LO * 65536)
        self.y_hi = int(Y_HI * 65536)
        self.t_lo = int(TURN_LO * 65536)
        self.t_hi = int(TURN_HI * 65536)
        k = (1 + SLOPE * SLOPE) ** 0.5
        self.nc = int(4096 / k)  # wall normal: horizontal part
        self.ns = int(4096 * SLOPE / k)  # and vertical part
        self.br3 = int(BALL_R * 1024)
        self.br2d = int(BALL_R2D * 1024)
        self.speed = Param(ctx, "speed", 1.0)
        self.skill = Param(ctx, "skill", 0.7, 256)
        self.style = Param(ctx, "style", 0, 1)
        self.skill_v = self.skill.get()
        self.v = SERVE_V
        self.t = 0  # game clock, ms at speed 1
        self.last = 0
        self.sub = 0
        self.predicting = 0
        self.pred_steps = 0
        self.scorer = 0
        self.lost = 0
        self.reset(1 if self.style.get() else 0)

    def reset(self, mode):
        """Start (or switch style): clear the trail, centre the paddles and
        serve after the usual pause."""
        self.mode = mode
        self.state = _MISS
        self.state_t = self.t - MISS_MS
        self.predicting = 0
        for s in range(2):
            self.pad_z[s] = 0 if mode else s << 15
            self.tgt_z[s] = self.pad_z[s]
            self.pad_y[s] = (self.y_lo + self.y_hi) >> 1
            self.tgt_y[s] = self.pad_y[s]
        for j in range(self.n_act):
            self.trail[self.act[j]] = 0
        self.n_act = 0

    # ─── Physics (integer only, no allocation) ───────────────────────────

    def move(self, s, dt):
        """Advance ball state s by dt ms. Returns 0, 1 for a plain wall, or
        2 / 3 at side 0's / side 1's goal (already reflected)."""
        if not self.mode:
            return self.move2d(s, dt)
        s[_X] += s[_VX] * dt
        s[_Y] += s[_VY] * dt
        s[_Z] += s[_VZ] * dt
        if s[_Y] < self.y_lo and s[_VY] < 0:
            s[_Y] = self.y_lo
            s[_VY] = -s[_VY]
        elif s[_Y] > self.y_hi and s[_VY] > 0:
            s[_Y] = self.y_hi
            s[_VY] = -s[_VY]
        x = s[_X] >> 4
        z = s[_Z] >> 4
        b = s[_Y] >> 11
        b = 0 if b < 0 else BINS - 1 if b >= BINS else b
        w = self.wall[b]
        r2 = x * x + z * z
        if r2 <= w * w:
            return 0
        r = isqrt(r2)
        nx = x * self.nc // r
        nz = z * self.nc // r
        ny = self.ns
        d = (s[_VX] * nx + s[_VY] * ny + s[_VZ] * nz) >> 12
        # Back onto the wall.
        s[_X] = s[_X] * w // r
        s[_Z] = s[_Z] * w // r
        if d <= 0:
            return 0
        s[_VX] -= (2 * d * nx) >> 12
        s[_VY] -= (2 * d * ny) >> 12
        s[_VZ] -= (2 * d * nz) >> 12
        ax = x if x >= 0 else -x
        az = z if z >= 0 else -z
        if ax >= az:
            return 2 if x > 0 else 3
        self.keep(s)
        return 1

    def move2d(self, s, dt):
        b = s[_Y] >> 11
        b = 0 if b < 0 else BINS - 1 if b >= BINS else b
        s[_X] = (s[_X] + s[_VX] * dt * 4096 // self.circ[b]) & 0xFFFF
        s[_Y] += s[_VY] * dt
        if s[_Y] < self.t_lo and s[_VY] < 0:
            s[_Y] = self.t_lo
            s[_VY] = -s[_VY]
            return 2
        if s[_Y] > self.t_hi and s[_VY] > 0:
            s[_Y] = self.t_hi
            s[_VY] = -s[_VY]
            return 3
        return 0

    def keep(self, s):
        """Keep enough speed along the goal axis (x, or y in style 0) that
        the ball crosses instead of rattling or circling."""
        k = _VX if self.mode else _VY
        need = self.v * MIN_ALONG >> 8
        v = s[k]
        if -need < v < need:
            s[k] = need if v >= 0 else -need

    def set_speed(self, s, v):
        m = isqrt(s[_VX] * s[_VX] + s[_VY] * s[_VY] + s[_VZ] * s[_VZ]) or 1
        s[_VX] = s[_VX] * v // m
        s[_VY] = s[_VY] * v // m
        s[_VZ] = s[_VZ] * v // m

    def serve(self):
        s = self.ball
        self.v = SERVE_V
        toward = 1 if self.scorer == 0 else -1  # towards the scorer
        r1 = self.rand(8) - 128
        r2 = self.rand(8) - 128
        if self.mode:
            s[_X] = 0
            s[_Z] = 0
            s[_Y] = (self.y_lo + self.y_hi) >> 1
            s[_VX] = toward * 200
            s[_VY] = r1 * 3 // 4
            s[_VZ] = r2
        else:
            s[_X] = self.rand(16)
            s[_Y] = (self.t_lo + self.t_hi) >> 1
            s[_Z] = 0
            s[_VY] = -toward * 160  # side 0 is the bottom
            s[_VX] = r1 * 2 if r1 else 100
            s[_VZ] = 0
        self.set_speed(s, self.v)
        self.keep(s)
        self.start_predict()

    def start_predict(self):
        p = self.pred
        s = self.ball
        for k in range(6):
            p[k] = s[k]
        self.predicting = 1
        self.pred_steps = 0

    def predict(self):
        """Run the predictor a slice further; when it reaches a goal, aim
        that side's paddle there with an error that grows as skill drops."""
        if not self.predicting:
            return
        p = self.pred
        for _ in range(PREDICT_STEPS):
            hit = self.move(p, STEP_MS)
            self.pred_steps += 1
            if hit >= 2:
                side = hit - 2
                err = (256 - self.skill_v) * 3
                ez = ((self.rand(9) - 256) * err) >> 8
                ey = ((self.rand(9) - 256) * err) >> 8
                if self.mode:
                    self.tgt_z[side] = p[_Z] + (ez * self.pad_w >> 8)
                    self.tgt_y[side] = p[_Y] + (ey * self.pad_h >> 8)
                    # The other paddle drifts back towards the middle.
                    self.tgt_z[1 - side] = 0
                    self.tgt_y[1 - side] = (self.y_lo + self.y_hi) >> 1
                else:
                    self.tgt_z[side] = (p[_X] + (ez * self.pad_a >> 8)) & 0xFFFF
                self.predicting = 0
                return
            if self.pred_steps > 1000:
                self.predicting = 0
                return

    def move_paddles(self, dt):
        if not self.mode:
            step = PADDLE_A * dt
            for s in range(2):
                d = ((self.tgt_z[s] - self.pad_z[s] + 32768) & 0xFFFF) - 32768
                d = step if d > step else -step if d < -step else d
                self.pad_z[s] = (self.pad_z[s] + d) & 0xFFFF
            return
        step = PADDLE_V * dt
        for s in range(2):
            d = self.tgt_z[s] - self.pad_z[s]
            self.pad_z[s] += step if d > step else -step if d < -step else d
            d = self.tgt_y[s] - self.pad_y[s]
            self.pad_y[s] += step if d > step else -step if d < -step else d

    def covered(self, side):
        """Is the ball, at side's goal, within reach of side's paddle? If
        so, steer it by where it struck."""
        s = self.ball
        if self.mode:
            oz = s[_Z] - self.pad_z[side]
            oy = s[_Y] - self.pad_y[side]
            wz = self.pad_w + self.reach
            wy = self.pad_h + self.reach
            if -wz < oz < wz and -wy < oy < wy:
                s[_VZ] += oz * self.v // wz
                s[_VY] += oy * self.v // wy
                return True
            return False
        # Style 0: angular offset; REACH converted to turns at this height.
        oa = ((s[_X] - self.pad_z[side] + 32768) & 0xFFFF) - 32768
        b = s[_Y] >> 11
        b = 0 if b < 0 else BINS - 1 if b >= BINS else b
        wa = self.pad_a + (self.reach >> 4) * 65536 // self.circ[b]
        if -wa < oa < wa:
            s[_VX] += oa * self.v // wa
            return True
        return False

    def step(self, dt):
        self.t += dt
        if self.state == _MISS:
            self.move_paddles(dt)
            if self.t - self.state_t >= MISS_MS + SERVE_MS:
                self.state = _PLAY
                self.serve()
            return
        while dt > 0:
            h = dt if dt < STEP_MS else STEP_MS
            dt -= h
            self.move_paddles(h)
            hit = self.move(self.ball, h)
            if hit < 2:
                continue
            side = hit - 2
            if self.covered(side):
                v = self.v + (self.v * SPEEDUP >> 8)
                self.v = v if v < MAX_V else MAX_V
                self.set_speed(self.ball, self.v)
                self.keep(self.ball)
                self.hit_t[side] = self.t
                self.start_predict()
            else:
                self.state = _MISS
                self.state_t = self.t
                self.scorer = 1 - side
                self.lost = side
                return

    # ─── Rendering ───────────────────────────────────────────────────────

    def render(self, t_ms, frame):
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.sub += dt * self.speed.get()
        self.skill_v = self.skill.get()
        mode = 1 if self.style.get() else 0
        if mode != self.mode:
            self.reset(mode)
        self.step(self.sub >> 12)
        self.sub &= 4095
        self.predict()

        # Ball fades out over a miss; the scorer's half flashes once.
        bamp = 256
        flash = 0
        if self.state == _MISS:
            e = self.t - self.state_t
            if e < MISS_MS:
                q = (MISS_MS - e) * 256 // MISS_MS
                flash = q * q >> 8
                bamp = q
            else:
                bamp = 0
        # Paddle brightness: brief boost on a return; the side that missed
        # goes dark with the flash.
        l0 = 150
        l1 = 150
        e = self.t - self.hit_t[0]
        if e < HIT_MS:
            l0 += 106 * (HIT_MS - e) // HIT_MS
        e = self.t - self.hit_t[1]
        if e < HIT_MS:
            l1 += 106 * (HIT_MS - e) // HIT_MS
        if flash:
            if self.lost == 0:
                l0 = l0 * (256 - flash) >> 8
            else:
                l1 = l1 * (256 - flash) >> 8

        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(self.bg)] = self.bg

        sc = self.scorer
        if flash:
            fl = self.flash_lv[2 * mode + sc] * flash >> 8
            r = SIDES[3 * sc] * fl >> 8
            g = SIDES[3 * sc + 1] * fl >> 8
            b = SIDES[3 * sc + 2] * fl >> 8
            lst = self.halves[mode][sc]
            for j in range(len(lst)):
                k = 3 * lst[j]
                frame[k] = r
                frame[k + 1] = g
                frame[k + 2] = b

        px = self.px
        py = self.py
        pz = self.pz
        pa = self.pa
        pw = self.pad_w >> 6
        ph = self.pad_h >> 6
        wa = self.pad_a
        for s in range(2):
            lv = l1 if s else l0
            r = SIDES[3 * s] * lv >> 8
            g = SIDES[3 * s + 1] * lv >> 8
            b = SIDES[3 * s + 2] * lv >> 8
            zc = self.pad_z[s]
            lst = self.pads[mode][s]
            if mode:
                zc >>= 6
                yc = self.pad_y[s] >> 6
                for j in range(len(lst)):
                    i = lst[j]
                    dz = pz[i] - zc
                    dy = py[i] - yc
                    if -pw < dz < pw and -ph < dy < ph:
                        k = 3 * i
                        frame[k] = r
                        frame[k + 1] = g
                        frame[k + 2] = b
            else:
                for j in range(len(lst)):
                    i = lst[j]
                    d = ((pa[i] - zc + 32768) & 0xFFFF) - 32768
                    if -wa < d < wa:
                        k = 3 * i
                        frame[k] = r
                        frame[k + 1] = g
                        frame[k + 2] = b

        # Trail: fade what's lit, dropping LEDs that reach zero; about a
        # 100 ms half-life.
        decay = 256 - dt * 2
        if decay < 0:
            decay = 0
        trail = self.trail
        act = self.act
        m = 0
        for j in range(self.n_act):
            i = act[j]
            v = trail[i] * decay >> 8
            trail[i] = v
            if v:
                act[m] = i
                m += 1
        # Ball: only the height bins it overlaps.
        if bamp:
            ball = self.ball
            by = ball[_Y] >> 6
            br = self.br3 if mode else self.br2d
            br2 = br * br
            b0 = (by - br) * YBINS >> 10
            b1 = (by + br) * YBINS >> 10
            b0 = 0 if b0 < 0 else b0
            b1 = YBINS - 1 if b1 > YBINS - 1 else b1
            bx = ball[_X] >> 6
            bz = ball[_Z] >> 6
            ba = ball[_X]
            # Style 0: arc length per turn at the ball's height.
            cb = ball[_Y] >> 11
            cb = 0 if cb < 0 else BINS - 1 if cb >= BINS else cb
            circ = self.circ[cb]
            order = self.yorder
            for j in range(self.ybin_start[b0], self.ybin_start[b1 + 1]):
                i = order[j]
                if mode:
                    dx = px[i] - bx
                    if not -br < dx < br:
                        continue
                    dz = pz[i] - bz
                    if not -br < dz < br:
                        continue
                    dy = py[i] - by
                    d2 = dx * dx + dy * dy + dz * dz
                else:
                    da = ((pa[i] - ba + 32768) & 0xFFFF) - 32768
                    dx = da * circ >> 18  # 1/1024 tree heights
                    if not -br < dx < br:
                        continue
                    dy = py[i] - by
                    d2 = dx * dx + dy * dy
                if d2 < br2:
                    q = (br2 - d2) * 255 // br2
                    q = q * bamp >> 8
                    v = trail[i]
                    if q > v:
                        if not v:
                            act[m] = i
                            m += 1
                        trail[i] = q
        self.n_act = m
        # Ball and trail over whatever is underneath.
        wr, wg, wb = BALL
        for j in range(m):
            i = act[j]
            v = trail[i]
            k = 3 * i
            r = frame[k]
            g = frame[k + 1]
            b = frame[k + 2]
            frame[k] = r + ((wr - r) * v >> 8)
            frame[k + 1] = g + ((wg - g) * v >> 8)
            frame[k + 2] = b + ((wb - b) * v >> 8)
