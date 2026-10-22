"""Portal: a box falls through a pair of portals, faster and faster.

A blue portal glows in a ring low on the tree and an orange one in a
ring higher up, each with a slow swirl. A box (a companion cube of
light, flat-lit with hard edges and a pink heart at its centre) drops out
of the orange portal, falls straight down the middle of the tree into the
blue one and comes straight back out of the orange one, keeping its
speed: one continuous fall that wraps through the portals, so every pass
is quicker than the last (the infinite-fall trick). It is clipped at the
portal planes, so while it is part-way through it shows in both places:
sinking into the blue portal and emerging from the orange one. As it
speeds up it is motion-blurred into a streak, and at top speed into a
steady beam between the portals. On its last pass into the blue portal a
wave of light rises up the tree from it, everything fades out, and after
a moment of dark the portals open again.

The portals and the fall share one vertical axis, fitted to the trunk
from the coordinates (circle centres of height slices), so the box falls
through the middle of both rings whatever the calibration's origin.

One build-up, climax and fade fits exactly in PERIOD_MS, which is also
DURATION_MS: an auto-cycle visit at default speed plays it once, a locked
effect repeats it with no visible restart, and the web export stores it
as an exact loop.
"""

import math
from array import array

from effects._output import budget_for_current
from effects._util import SIN, Param, Phase

NAME = "portal"
PERIOD_MS = 19000  # one cycle at speed 1
DURATION_MS = PERIOD_MS
PARAMS = {
    "speed": (1.0, 0.5, 2.0),
}
NEEDS_COORDS = True

# Timeline, ms at speed 1.
OPEN_MS = 700  # portals fade in
FALL_MS = 16200  # the whole accelerating fall
# Passes per second through the portals at points of the fall (ms into
# it), interpolated exponentially: most of the time goes to the build, and
# the last two seconds are much the fastest. Scaled so the fall is a whole
# number of passes.
RATE = ((0, 0.8), (4000, 1.05), (8000, 1.45), (11000, 1.95), (13000, 2.45),
        (14500, 3.0), (15300, 4.0), (15800, 5.5), (FALL_MS, 9.0))
STEPS = 512  # distance table resolution over the fall
DIP_MS = 250  # portals dim just before the end of the fall, for contrast
WAVE_MS = 1000  # the closing wave climbs from the blue portal to the top
ATTACK_MS = 160  # wave rise
FADE_MS = 900  # portals fade out after the fall

# Geometry, tree heights.
AXIS_FROM = 0.1  # fit the trunk axis to slices above this (the base is ragged)
AXIS_SLICE = 0.05
BLUE_BAND = (0.22, 0.32)  # portal rings: LEDs in this height band...
ORANGE_BAND = (0.62, 0.74)
RING_RAD = ((0.7, 1.35), (0.7, 1.35))  # ...at this share of its median radius (blue, orange)
PATH_D = 0.05  # the box falls this far in front of the axis (+z)
BOX = 0.075  # cube half-size
CORE = 0.035  # the heart: pink inside this half-size...
CORE_EDGE = 0.025  # ...blending to the cube's white over this
EDGE = 0.02  # soft edge outside the cube
WAVE_W = 0.12  # half-width of the closing wave

# Motion blur. The streak behind the cube grows as the pass rate to the
# fourth and fills the whole column at BEAM_RATE, so from there on every LED
# between the portals is steadily lit (a beam) instead of blinking at the
# pass rate. The streak is drawn at 1 - HEAD of the cube's level, and the
# cube shows on top of it, fading out above HEAD_RATE as
# (HEAD_RATE / rate)^6: the cube stays visible as it speeds into the beam,
# no LED is modulated by more than about a tenth above 4 passes a second,
# and the beam brightens to the cube's level as the cube dissolves into it.
BEAM_RATE = 3.0
HEAD = 0.55
HEAD_RATE = 3.0

BLUE = (0, 110, 255)
ORANGE = (255, 45, 0)
CUBE = (215, 222, 235)
PINK = (255, 25, 110)
HEART_RATE = (1.2, 2.0)  # the heart fades to white between these passes a second
WAVE = (200, 225, 255)
PORTAL = 110  # /256 portal glow; the swirl adds up to SWIRL_AMP
SWIRL_AMP = 70
SWIRL_MS = 2400  # one turn of the swirl at rest...
SWIRL_K = 0.5  # ...and SWIRL_K turns per pass during the fall,
SWIRL_MAX = 2.0  # up to this many turns a second
DIP = 110  # /256 the portals dim to at the end of the fall
SHARE = 0.92  # beam and wave peaks: this share of the power budget

F = 1024  # 1/1024 tree heights, the integer unit of render()


def _axis(pts):
    """Fit the trunk axis: centre x, z = (x0 + kx h, z0 + kz h), from the
    algebraic circle fit of each height slice, weighted by its LED count."""
    rows = []
    h0 = AXIS_FROM
    while h0 < 1.0:
        sx = sz = 0.0
        n = 0
        sel = []
        for p in pts:
            if h0 <= p[1] < h0 + AXIS_SLICE:
                sel.append(p)
                sx += p[0]
                sz += p[2]
                n += 1
        if n >= 6:
            mx = sx / n
            mz = sz / n
            a = b = c = d = e = 0.0
            for p in sel:
                x = p[0] - mx
                z = p[2] - mz
                w = x * x + z * z
                a += x * x
                b += x * z
                c += z * z
                d += x * w
                e += z * w
            det = a * c - b * b
            if det > 1e-12:
                # Centre offset solves [a b; b c] [2u 2v] = [d e].
                u = (d * c - b * e) / det / 2
                v = (a * e - b * d) / det / 2
                rows.append((h0 + AXIS_SLICE / 2, mx + u, mz + v, n))
        h0 += AXIS_SLICE
    if len(rows) < 2:
        return 0.0, 0.0, 0.0, 0.0
    sw = sh = shh = 0.0
    for h, _, _, n in rows:
        sw += n
        sh += n * h
        shh += n * h * h
    mh = sh / sw
    var = shh / sw - mh * mh
    out = []
    for k in (1, 2):
        sv = shv = 0.0
        for row in rows:
            sv += row[3] * row[k]
            shv += row[3] * row[0] * row[k]
        mv = sv / sw
        slope = (shv / sw - mh * mv) / var if var > 1e-9 else 0.0
        out.append(mv - slope * mh)
        out.append(slope)
    return out[0], out[1], out[2], out[3]


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        coords = ctx.coords or []
        pts = []
        known = bytearray(n)
        for i in range(n):
            c = coords[i] if i < len(coords) else None
            if c is not None:
                known[i] = 1
                pts.append(c)
        x0, kx, z0, kz = _axis(pts)
        self.axis = (x0, kx, z0, kz)
        pts = None
        # Offsets from the axis at each LED's own height.
        dxs = [0.0] * n
        dzs = [0.0] * n
        for i in range(n):
            if known[i]:
                x, y, z = coords[i]
                dxs[i] = x - (x0 + kx * y)
                dzs[i] = z - (z0 + kz * y)

        # Portal rings: a height band, and within it the LEDs near the
        # band's median distance from the axis. Each LED's angle round the
        # axis (0..255) drives the swirl.
        rings = []
        for (lo, hi), (rlo, rhi) in zip((BLUE_BAND, ORANGE_BAND), RING_RAD):
            band = []
            for i in range(n):
                if known[i] and lo <= coords[i][1] < hi:
                    band.append((math.sqrt(dxs[i] * dxs[i] + dzs[i] * dzs[i]), i))
            band.sort()
            med = band[len(band) // 2][0] if band else 0.0
            rings.append(array("H", sorted(i for r, i in band if rlo * med <= r <= rhi * med)))
            band = None
        self.rings = (rings[0], rings[1])
        self.ang = bytearray(n)
        for i in range(n):
            a = math.atan2(dzs[i], dxs[i]) / (2 * math.pi)
            self.ang[i] = int(a * 256) & 255

        # The fall runs from the middle of the orange band to the middle of
        # the blue one, along the axis, PATH_D in front of it. LEDs within
        # the cube's reach of that line, with their offsets.
        self.y_top = int((ORANGE_BAND[0] + ORANGE_BAND[1]) / 2 * F)
        self.y_bot = int((BLUE_BAND[0] + BLUE_BAND[1]) / 2 * F)
        reach = BOX + EDGE
        path = []
        for i in range(n):
            if not known[i]:
                continue
            y = int(coords[i][1] * F)
            ox = dxs[i]
            oz = dzs[i] - PATH_D
            if self.y_bot <= y <= self.y_top and -reach < ox < reach and -reach < oz < reach:
                ax = ox if ox >= 0 else -ox
                az = oz if oz >= 0 else -oz
                path.append((i, y, int((ax if ax > az else az) * F)))
        self.path = array("H", [p[0] for p in path])
        self.py = array("h", [p[1] for p in path])
        self.pm = array("h", [p[2] for p in path])  # Chebyshev offset
        dxs = dzs = None

        span = self.y_top - self.y_bot  # one pass, 1/1024 tree heights
        self.span = span
        # Distance fallen (and swirl turned, /256 turns) at each of STEPS + 1
        # points of the fall, from the pass rate keyframes.
        cum = [0.0] * (STEPS + 1)
        swc = [0.0] * (STEPS + 1)
        dt = FALL_MS / STEPS
        for k in range(STEPS):
            tm = (k + 0.5) * dt
            j = 0
            while RATE[j + 1][0] < tm:
                j += 1
            (t0, r0), (t1, r1) = RATE[j], RATE[j + 1]
            r = r0 * (r1 / r0) ** ((tm - t0) / (t1 - t0))
            cum[k + 1] = cum[k] + r * dt / 1000
            sw = SWIRL_K * r
            sw = 1000 / SWIRL_MS if sw < 1000 / SWIRL_MS else SWIRL_MAX if sw > SWIRL_MAX else sw
            swc[k + 1] = swc[k] + sw * dt * 256 / 1000
        passes = max(1, int(cum[STEPS] + 0.5))
        self.passes = passes
        scale = 16 * passes * span / cum[STEPS]
        self.dist = array("i", [int(c * scale + 0.5) for c in cum])  # 1/16 units
        self.swf = array("H", [int(c + 0.5) for c in swc])
        cum = swc = None
        self.sw_open = OPEN_MS * 256 // SWIRL_MS
        self.sw_end = self.sw_open + self.swf[STEPS]
        self.v_end = (self.dist[STEPS] - self.dist[STEPS - 1]) * STEPS * 1000 // (16 * FALL_MS)
        self.v_beam = int(BEAM_RATE * span)  # units/s
        self.v_head = int(HEAD_RATE * span)
        self.v_heart = (int(HEART_RATE[0] * span), int(HEART_RATE[1] * span))
        self.core = int(CORE * F)
        self.core_e = int(CORE_EDGE * F)
        self.head = int(HEAD * 256)
        self.box = int(BOX * F)
        self.edge = int(EDGE * F)
        self.s2 = 2 * self.box + self.edge  # integral of the cube's profile

        # Power: the beam lights the whole column (the cube itself, only a
        # few LEDs, goes to full on top of the streak); cap it, with the portals
        # at their brightest, at SHARE of the budget. The wave is capped the
        # same way at its fullest band.
        budget = budget_for_current() * SHARE
        portals = (len(rings[0]) * sum(BLUE) + len(rings[1]) * sum(ORANGE)) \
            * (PORTAL + SWIRL_AMP) // 256
        col = len(path) * sum(CUBE)
        self.cube_max = max(0, min(256, int(256 * (budget - portals) / col))) if col else 256
        path = None

        top = 1.0 - (BLUE_BAND[0] + BLUE_BAND[1]) / 2
        self.wave_reach = int((top + WAVE_W) * F)
        self.wd = array("H", bytes(2 * n))  # height distance from the blue portal
        bins = [0] * 66
        for i in range(n):
            if known[i]:
                d = coords[i][1] * F - self.y_bot
                d = int(d if d >= 0 else -d)
                self.wd[i] = d
                bins[min(65, d >> 4)] += 1
        ww = int(WAVE_W * F)
        self.ww = ww
        worst = 1
        for rho in range(0, self.wave_reach, 16):
            s = 0
            for b in range(66):
                g = ww - abs(b * 16 + 8 - rho)
                if g > 0:
                    s += bins[b] * g
            worst = max(worst, s // ww)
        self.wave_max = min(256, int(256 * budget / (worst * sum(WAVE))))

        self.zero = bytearray(3 * n)
        self.known = known
        self.frame = None
        self.view = None
        self.speed = Param(ctx, "speed", 1.0)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        self.phase.update(t_ms, self.speed.get())
        t = self.phase.acc >> 12  # ms into the cycle at speed 1, exact
        if frame is not self.frame:
            self.frame = frame
            self.view = memoryview(frame)
        self.view[0:len(self.zero)] = self.zero

        # Portal level: open, hold, dip at the end of the fall, fade, rest.
        ft = t - OPEN_MS - FALL_MS
        if t < OPEN_MS:
            master = 256 * t // OPEN_MS
        elif ft < -DIP_MS:
            master = 256
        elif ft < 0:
            master = 256 - (256 - DIP) * (ft + DIP_MS) // DIP_MS
        elif ft < FADE_MS:
            q = (FADE_MS - ft) * 256 // FADE_MS
            master = DIP * (q * q >> 8) >> 8
        else:
            master = 0
        if master:
            # Swirl: steady while the portals open and close, speeding up
            # with the fall in between.
            tau = t - OPEN_MS
            if tau < 0:
                sw = t * 256 // SWIRL_MS
            elif tau < FALL_MS:
                n = tau * STEPS
                k = n // FALL_MS
                sw = self.swf[k + 1] - self.swf[k]
                sw = self.sw_open + self.swf[k] + sw * (n - k * FALL_MS) // FALL_MS
            else:
                sw = self.sw_end + (tau - FALL_MS) * 256 // SWIRL_MS
            ang = self.ang
            for side in range(2):
                r, g, b = ORANGE if side else BLUE
                lst = self.rings[side]
                for j in range(len(lst)):
                    i = lst[j]
                    a = (ang[i] + sw) & 255 if side else (ang[i] - sw) & 255
                    lv = (PORTAL + (SIN[a] * SWIRL_AMP >> 8)) * master >> 8
                    k = 3 * i
                    frame[k] = r * lv >> 8
                    frame[k + 1] = g * lv >> 8
                    frame[k + 2] = b * lv >> 8

        # The fall, from the distance table, wrapping through the portals,
        # so speed carries over from pass to pass. After the last pass the
        # streak drains into the blue portal at the final speed.
        tau = t - OPEN_MS
        if 0 <= tau < FALL_MS:
            n = tau * STEPS
            k = n // FALL_MS
            dd = self.dist[k + 1] - self.dist[k]
            dist = (self.dist[k] + dd * (n - k * FALL_MS) // FALL_MS) >> 4
            v = dd * (STEPS * 1000 // 16) // FALL_MS  # 1/1024 tree heights a second
            p = dist // self.span  # passes completed
            self.fall(self.y_top - (dist - p * self.span), v, p, p < self.passes - 1, frame)
        elif 0 <= tau - FALL_MS < 200:
            yb = self.y_bot - self.v_end * (tau - FALL_MS) // 1000
            if yb + self.span + self.box + self.edge > self.y_bot:
                self.fall(yb, self.v_end, self.passes - 1, False, frame)

        # The closing wave: climbs from the blue portal as the box drops
        # into it for good; each LED sees it pass once.
        if 0 <= ft < WAVE_MS:
            rho = self.wave_reach * ft // WAVE_MS
            if ft < ATTACK_MS:
                amp = self.wave_max * ft // ATTACK_MS
            else:
                q = (WAVE_MS - ft) * 256 // (WAVE_MS - ATTACK_MS)
                amp = self.wave_max * (q * q >> 8) >> 8
            ww = self.ww
            wd = self.wd
            known = self.known
            wr, wg, wb = WAVE
            for i in range(self.n):
                if known[i]:
                    d = wd[i] - rho
                    d = d if d >= 0 else -d
                    if d < ww:
                        lv = amp * (ww - d) // ww
                        k = 3 * i
                        v = frame[k] + (wr * lv >> 8)
                        frame[k] = v if v < 255 else 255
                        v = frame[k + 1] + (wg * lv >> 8)
                        frame[k + 1] = v if v < 255 else 255
                        v = frame[k + 2] + (wb * lv >> 8)
                        frame[k + 2] = v if v < 255 else 255

    def fall(self, yb, v, p, more, frame):
        """Draw the cube centred at height yb (1/1024 tree heights), falling
        at v (1/1024 tree heights a second) after p whole passes, with its
        streak, and its copies one and two passes behind (still sinking into
        the blue portal) and, if `more`, one ahead (already out of the
        orange one)."""
        s = self.box
        e = self.edge
        se = s + e
        s2 = self.s2
        span = self.span
        # Streak length: grows as the pass rate to the fourth, the whole
        # column at BEAM_RATE. Cube weight: HEAD, fading as
        # (HEAD_RATE / rate)^6.
        x = v * 256 // self.v_beam
        if x >= 256:
            lt = span
        else:
            x = x * x >> 8
            lt = span * (x * x >> 8) >> 8
        lt = lt if lt > 0 else 1
        den = lt if lt < s2 else s2
        w = self.head
        if v > self.v_head:
            x = self.v_head * 256 // v
            x = x * x * x >> 16
            w = w * x * x >> 16
        cmax = self.cube_max
        # Copies at yb - m span: m = -1 is the next pass, m = 1, 2 earlier.
        m_lo = -1 if more else 0
        m_hi = p if p < 2 else 2
        cr, cg, cb = CUBE
        path = self.path
        py = self.py
        pm = self.pm
        # Heart: the cube's core is pink, clipped at the portals like the
        # rest of it; it fades to white as the cube speeds up, so each pass
        # stays one flash for any LED.
        v0, v1 = self.v_heart
        hf = 256 if v <= v0 else 0 if v >= v1 else (v1 - v) * 256 // (v1 - v0)
        co = self.core
        ce = self.core_e
        pr_, pg, pb = PINK
        for j in range(len(path)):
            hm = pm[j]
            if hm >= se:
                continue
            # Streak: (P(d) - P(d - L)) / L, the cube's trapezoid profile
            # (1 inside s, ramping to 0 over e) averaged over its last L of
            # travel, normalised to 1 where the streak is solid. Cube: the
            # profile itself.
            cov = 0
            hd = 0
            ph = 0
            d = py[j] - yb + m_lo * span
            for m in range(m_lo, m_hi + 1):
                if -se < d < lt + se:
                    if d >= se and d - lt <= -se:
                        cov += s2
                    else:
                        cov += _prim(d, s, e, se) - _prim(d - lt, s, e, se)
                    if d < se:
                        ad = d if d >= 0 else -d
                        hd += 256 if ad <= s else (se - ad) * 256 // e
                        if hf:
                            c3 = ad if ad > hm else hm
                            if c3 < co + ce:
                                ph += 256 if c3 <= co else (co + ce - c3) * 256 // ce
                d += span
            if cov <= 0:
                continue
            cov = cov * 256 // den
            cov = cov if cov < 256 else 256
            hd = hd if hd < 256 else 256
            lv = (cov * (256 - w) * cmax >> 16) + (hd * w >> 8)
            if hm > s:
                lv = lv * (se - hm) // e
            i = path[j]
            k = 3 * i
            if ph:
                ph = (ph if ph < 256 else 256) * hf >> 8
                q = 256 - ph
                r = (cr * q + pr_ * ph) >> 8
                g = (cg * q + pg * ph) >> 8
                b = (cb * q + pb * ph) >> 8
            else:
                r, g, b = cr, cg, cb
            c = frame[k] + (r * lv >> 8)
            frame[k] = c if c < 255 else 255
            c = frame[k + 1] + (g * lv >> 8)
            frame[k + 1] = c if c < 255 else 255
            c = frame[k + 2] + (b * lv >> 8)
            frame[k + 2] = c if c < 255 else 255

def _prim(u, s, e, se):
    """Integral up to u of the cube's vertical profile: 1 for |u| < s,
    ramping linearly to 0 at |u| = s + e."""
    if u <= -se:
        return 0
    if u < -s:
        w = u + se
        return w * w // (2 * e)
    if u < s:
        return (e >> 1) + u + s
    if u < se:
        w = se - u
        return 2 * s + e - w * w // (2 * e)
    return 2 * s + e
