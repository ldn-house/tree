"""Portal: a box falls through a pair of portals, faster and faster.

A blue portal glows in a ring at the base of the tree and an orange one
in a band near the top, each with a slow swirl. A warm-white box falls
down the front of the tree from the orange portal into the blue one and
comes straight back out of the orange one, keeping its speed, so each
fall is shorter than the last: the infinite-fall trick. When it is going
too fast to follow it becomes a speed blur, a streak the length of the
fall that brightens, then the whole tree gives one flash, everything
fades out, and after a moment of dark the portals open again.

One build-up, climax and fade fits exactly in PERIOD_MS, which is also
DURATION_MS: an auto-cycle visit at default speed plays it once, a locked
effect repeats it with no visible restart, and the web export stores it
as an exact loop.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, SIN, Layout, Param, Phase

NAME = "portal"
PERIOD_MS = 19000  # one cycle at speed 1
DURATION_MS = PERIOD_MS
PARAMS = {
    "speed": (1.0, 0.5, 2.0),
}
NEEDS_COORDS = True

# Timeline, ms at speed 1.
OPEN_MS = 1200  # portals fade in
FALL0_MS = 2400  # first fall
FALL_RATIO = 210  # /256: each fall this much of the one before (~0.82)
FALL_MIN_MS = 140  # falls stop being drawn as a box below this
BLUR_FROM_MS = 420  # the box starts smearing into a streak below this
BLUR_MS = 2000  # the streak brightens for this long before the flash
ATTACK_MS = 150  # flash rise
FLASH_MS = 1400  # flash settle
FADE_MS = 1300  # portals and streak fade out after the flash starts

# Geometry, tree heights.
Y_TOP = 0.78  # the box leaves the orange portal here
Y_BOT = 0.05  # and enters the blue one here
BAND_BOT = 0.1  # blue portal: LEDs below this height
BAND_TOP = (0.72, 0.85)  # orange portal: LEDs in this band
PATH_R = 0.7  # the box falls at this share of the cone's radius, in front
BOX_R = 0.1  # box glow radius
STREAK_R = 0.08  # speed-blur half-width

BLUE = (0, 110, 255)
ORANGE = (255, 100, 0)
BOX = (255, 225, 190)
STREAK = (255, 240, 220)
FLASH = (255, 235, 210)
PORTAL = 170  # /256 portal glow; the swirl adds up to SWIRL_AMP
SWIRL_AMP = 86
SWIRL_MS = 2400  # one turn of the swirl
PEAK_SHARE = 1.1  # flash peak: just over the power budget
YBINS = 32
RBINS = 16  # radius profile bins


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        known = lay.known
        rm = lay.rmax
        count = lay.count or n
        # Portal bands, with each LED's angle for the swirl (0..255).
        bot = []
        top = []
        for i in range(n):
            if not known[i]:
                continue
            h = lay.h[i] / ONE
            if h < BAND_BOT:
                bot.append(i)
            elif BAND_TOP[0] <= h < BAND_TOP[1]:
                top.append(i)
        self.bands = (array("H", bot), array("H", top))
        bot = top = None
        self.ang = bytearray(n)
        for i in range(n):
            self.ang[i] = lay.theta[i] >> 4
        # The fall path: x = 0, z = PATH_R x the cone's local radius (the
        # front of the tree, towards the camera). Local radius: 85th
        # percentile per height bin, in 1/1024 tree heights.
        radii = [[] for _ in range(RBINS)]
        for i in range(n):
            if known[i]:
                radii[min(RBINS - 1, lay.h[i] * RBINS >> 12)].append(lay.r[i] * rm / 4)
        prof = [0] * RBINS
        last = 300
        for b in range(RBINS):
            rs = radii[b]
            if rs:
                rs.sort()
                last = rs[int(0.85 * (len(rs) - 1))]
            prof[b] = last
        radii = None
        # LEDs near the path, sorted by height into YBINS bins, with their
        # squared horizontal distance from it (1/1024 tree heights).
        reach = int(max(BOX_R, STREAK_R) * 1024)
        near = []
        for i in range(n):
            if not known[i]:
                continue
            y = lay.h[i] >> 2
            x = lay.x[i] * rm / 4
            z = lay.z[i] * rm / 4
            zp = PATH_R * prof[min(RBINS - 1, lay.h[i] * RBINS >> 12)]
            d2 = int(x * x + (z - zp) * (z - zp))
            if d2 < reach * reach:
                near.append((y, i, d2))
        near.sort()
        m = len(near)
        self.path = array("H", [e[1] for e in near])
        self.py = array("H", [e[0] for e in near])
        self.pd2 = array("H", [min(65535, e[2]) for e in near])
        self.bin_start = array("H", bytes(2 * (YBINS + 1)))
        j = 0
        for b in range(YBINS + 1):
            while j < m and near[j][0] * YBINS >> 10 < b:
                j += 1
            self.bin_start[b] = j
        near = None
        # The falls: start times and durations, each FALL_RATIO of the one
        # before, until they are too fast to draw.
        starts = []
        durs = []
        t = OPEN_MS
        d = FALL0_MS
        while d >= FALL_MIN_MS:
            starts.append(t)
            durs.append(d)
            t += d
            d = d * FALL_RATIO >> 8
        self.fall_t = array("i", starts)
        self.fall_d = array("i", durs)
        self.falls_end = t
        self.flash_t = t + BLUR_MS
        # Flash level so the whole tree peaks just over the budget.
        budget = budget_for_current()
        self.peak_sum = int(PEAK_SHARE * budget / count)
        self.y_top = int(Y_TOP * 1024)
        self.y_bot = int(Y_BOT * 1024)
        self.box_r = int(BOX_R * 1024)
        self.streak_r = int(STREAK_R * 1024)
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

        # Master level for portals and streak: open, hold, fade after the
        # flash, dark rest.
        ft = t - self.flash_t
        if t < OPEN_MS:
            master = 256 * t // OPEN_MS
        elif ft < 0:
            master = 256
        elif ft < FADE_MS:
            q = (FADE_MS - ft) * 256 // FADE_MS
            master = q * q >> 8
        else:
            master = 0
        # Find the current fall (if any) and how smeared it is.
        box_y = -1
        blur = 0
        nf = len(self.fall_t)
        if OPEN_MS <= t < self.falls_end:
            k = 0
            while k < nf - 1 and t >= self.fall_t[k + 1]:
                k += 1
            d = self.fall_d[k]
            s = (t - self.fall_t[k]) * 256 // d  # 0..255 through the fall
            box_y = self.y_top - ((self.y_top - self.y_bot) * (s * s >> 8) >> 8)
            if d < BLUR_FROM_MS:
                blur = (BLUR_FROM_MS - d) * 256 // (BLUR_FROM_MS - FALL_MIN_MS)
        elif self.falls_end <= t < self.flash_t + FADE_MS:
            blur = 256
        # The streak brightens through the blur phase; the portals swell.
        streak = 0
        swell = 0
        if blur:
            streak = blur * 150 >> 8
            if t >= self.falls_end:
                e = t - self.falls_end
                if e > BLUR_MS:
                    e = BLUR_MS
                streak += 106 * e // BLUR_MS
                swell = 60 * e // BLUR_MS
            streak = streak * master >> 8

        # Portals with a slow swirl, blue turning one way, orange the other.
        if master:
            sw = t * 256 // SWIRL_MS
            ang = self.ang
            for side in range(2):
                r, g, b = ORANGE if side else BLUE
                lst = self.bands[side]
                for j in range(len(lst)):
                    i = lst[j]
                    a = (ang[i] + sw) & 255 if side else (ang[i] - sw) & 255
                    lv = PORTAL + swell + (SIN[a] * SWIRL_AMP >> 8)
                    lv = (lv if lv < 256 else 256) * master >> 8
                    k = 3 * i
                    frame[k] = r * lv >> 8
                    frame[k + 1] = g * lv >> 8
                    frame[k + 2] = b * lv >> 8

        path = self.path
        py = self.py
        pd2 = self.pd2
        bs = self.bin_start
        # Speed blur: the whole fall, brightest along the path.
        if streak:
            r, g, b = STREAK
            w = self.streak_r
            w2 = w * w
            b0 = self.y_bot * YBINS >> 10
            b1 = self.y_top * YBINS >> 10
            for j in range(bs[b0], bs[b1 + 1]):
                d2 = pd2[j]
                if d2 < w2:
                    q = (w2 - d2) * streak // w2
                    self.add(path[j], r * q >> 8, g * q >> 8, b * q >> 8, frame)
        # The box, fading as it smears.
        if box_y >= 0 and blur < 256:
            amp = 256 - blur
            r, g, b = BOX
            w = self.box_r
            w2 = w * w
            b0 = (box_y - w) * YBINS >> 10
            b1 = (box_y + w) * YBINS >> 10
            b0 = 0 if b0 < 0 else b0
            b1 = YBINS - 1 if b1 > YBINS - 1 else b1
            for j in range(bs[b0], bs[b1 + 1]):
                dy = py[j] - box_y
                d2 = pd2[j] + dy * dy
                if d2 < w2:
                    q = (w2 - d2) * amp // w2
                    self.add(path[j], r * q >> 8, g * q >> 8, b * q >> 8, frame)
        # The flash: one, over everything, peaking just over the budget.
        if 0 <= ft < ATTACK_MS + FLASH_MS:
            if ft < ATTACK_MS:
                q = 256 * ft // ATTACK_MS
            else:
                q = (ATTACK_MS + FLASH_MS - ft) * 256 // FLASH_MS
                q = q * q >> 8
            r, g, b = FLASH
            lv = self.peak_sum * q // (r + g + b)
            lv = lv if lv < 256 else 256
            fr = r * lv >> 8
            fg = g * lv >> 8
            fb = b * lv >> 8
            known = self.known
            for i in range(self.n):
                if known[i]:
                    k = 3 * i
                    v = frame[k] + fr
                    frame[k] = v if v < 255 else 255
                    v = frame[k + 1] + fg
                    frame[k + 1] = v if v < 255 else 255
                    v = frame[k + 2] + fb
                    frame[k + 2] = v if v < 255 else 255

    def add(self, i, r, g, b, frame):
        k = 3 * i
        v = frame[k] + r
        frame[k] = v if v < 255 else 255
        v = frame[k + 1] + g
        frame[k + 1] = v if v < 255 else 255
        v = frame[k + 2] + b
        frame[k + 2] = v if v < 255 else 255
