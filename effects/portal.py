"""Portal: a box falls through a pair of portals, faster and faster.

A blue portal glows in a ring round the base of the tree and an orange
one in a ring higher up, each with a slow swirl. A box (a cube of light,
flat-lit with hard edges) drops out of the orange portal, falls down the
front of the tree into the blue one and comes straight back out of the
orange one, keeping its speed: one continuous fall under gravity that
wraps through the portals, so every pass is quicker than the last (the
infinite-fall trick). It is clipped at the portal planes, so it sinks
into one and emerges from the other, and a motion trail grows behind it
as it speeds up. On its last pass into the blue portal the tree gives
one flash, everything fades out, and after a moment of dark the portals
open again.

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
FALL_MS = 13000  # the whole accelerating fall
PASSES = 17  # times through the portals (the last pass is ~0.4 s)
ATTACK_MS = 150  # flash rise
FLASH_MS = 1400  # flash settle
DIP_MS = 350  # portals dim just before the flash, for contrast
FADE_MS = 1300  # portals fade out after the flash starts

# Geometry, tree heights.
BLUE_BAND = 0.1  # blue portal: LEDs below this height
ORANGE_BAND = (0.68, 0.76)  # orange portal: outer LEDs in this band
RING_R = 0.5  # portal rings use LEDs beyond this share of the local radius
Y_TOP = 0.72  # the box emerges from the orange portal here
Y_BOT = 0.05  # and sinks into the blue one here
PATH_R = 0.7  # the box falls at this share of the cone's radius, in front
BOX = 0.075  # cube half-size
EDGE = 0.02  # soft edge outside the cube
EXPOSURE_MS = 90  # trail length = distance fallen in this long

BLUE = (0, 110, 255)
ORANGE = (255, 45, 0)
CUBE = (215, 222, 235)
FLASH = (255, 235, 210)
PORTAL = 110  # /256 portal glow; the swirl adds up to SWIRL_AMP
SWIRL_AMP = 70
SWIRL_MS = 2400  # one turn of the swirl
TRAIL_LV = 140  # /256 at the head of the trail, fading to nothing
DIP = 90  # /256 the portals dim to before the flash
PEAK_SHARE = 0.95  # flash peak: about the power budget
RBINS = 16  # radius profile bins


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        known = lay.known
        rm = lay.rmax
        count = lay.count or n
        # Local cone radius (85th percentile per height bin), tree heights.
        radii = [[] for _ in range(RBINS)]
        for i in range(n):
            if known[i]:
                radii[min(RBINS - 1, lay.h[i] * RBINS >> 12)].append(lay.r[i] * rm / ONE)
        prof = [0.0] * RBINS
        last = 0.3
        for b in range(RBINS):
            rs = radii[b]
            if rs:
                rs.sort()
                last = rs[int(0.85 * (len(rs) - 1))]
            prof[b] = last
        radii = None
        # Portal rings, with each LED's angle for the swirl (0..255).
        blue = []
        orange = []
        for i in range(n):
            if not known[i]:
                continue
            h = lay.h[i] / ONE
            r = lay.r[i] * rm / ONE
            if h < BLUE_BAND:
                blue.append(i)
            elif ORANGE_BAND[0] <= h < ORANGE_BAND[1]:
                if r >= RING_R * prof[min(RBINS - 1, lay.h[i] * RBINS >> 12)]:
                    orange.append(i)
        self.rings = (array("H", blue), array("H", orange))
        blue = orange = None
        self.ang = bytearray(n)
        for i in range(n):
            self.ang[i] = lay.theta[i] >> 4
        # LEDs within the cube's reach of the fall path (x = 0, z = PATH_R
        # x the local radius, the front of the tree), with their offsets
        # from it in 1/1024 tree heights.
        reach = (BOX + EDGE) * 1024
        sel = []
        for i in range(n):
            if not known[i]:
                continue
            y = lay.h[i] >> 2
            x = lay.x[i] * rm / 4
            z = lay.z[i] * rm / 4 - PATH_R * prof[min(RBINS - 1, lay.h[i] * RBINS >> 12)] * 1024
            if -reach < x < reach and -reach < z < reach:
                sel.append(i)
                sel.append(y)
                sel.append(int(x))
                sel.append(int(z))
        self.path = array("H", sel[0::4])
        self.py = array("h", sel[1::4])
        self.dx = array("h", sel[2::4])
        self.dz = array("h", sel[3::4])
        sel = None
        self.y_top = int(Y_TOP * 1024)
        self.y_bot = int(Y_BOT * 1024)
        self.span = self.y_top - self.y_bot  # one pass, 1/1024 tree heights
        self.total = PASSES * self.span  # distance fallen in FALL_MS
        # Trail length = speed x EXPOSURE_MS; speed = 2 total t / FALL_MS^2.
        self.trail_k = 2 * self.total * EXPOSURE_MS // FALL_MS
        self.box = int(BOX * 1024)
        self.edge = int(EDGE * 1024)
        self.flash_t = OPEN_MS + FALL_MS
        budget = budget_for_current()
        self.peak_sum = int(PEAK_SHARE * budget / count)
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

        # Portal level: open, hold, dip before the flash, fade, dark rest.
        ft = t - self.flash_t
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
            sw = t * 256 // SWIRL_MS
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

        # The fall: distance under constant gravity, from rest, wrapping
        # through the portals, so speed carries over from pass to pass.
        tau = t - OPEN_MS
        if 0 <= tau < FALL_MS:
            dist = (tau * tau // FALL_MS) * self.total // FALL_MS
            trail = self.trail_k * tau // FALL_MS
            p = dist // self.span  # passes completed
            yb = self.y_top - (dist - p * self.span)
            # Draw the box where it is, the part already through the blue
            # portal (emerging below the orange one), and the part still
            # coming out of the orange portal (the tail of the last pass).
            self.cube(yb, trail, frame)
            if p < PASSES - 1:
                self.cube(yb + self.span, trail, frame)
            if p > 0:
                self.cube(yb - self.span, trail, frame)

        # The flash: one, as the box drops into the blue portal for good.
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

    def cube(self, yb, trail, frame):
        """Draw the cube centred at height yb (1/1024 tree heights), with a
        trail above it, clipped to the space between the portals."""
        top = self.y_top
        bot = self.y_bot
        if yb + self.box + trail < bot or yb - self.box > top:
            return
        s = self.box
        e = self.edge
        se = s + e
        cr, cg, cb = CUBE
        path = self.path
        py = self.py
        dx = self.dx
        dz = self.dz
        for j in range(len(path)):
            y = py[j]
            if y < bot or y > top:
                continue
            # Chebyshev distance: the largest of the axis offsets, so the
            # lit region is a cube rather than a ball.
            ax = dx[j]
            ax = ax if ax >= 0 else -ax
            az = dz[j]
            az = az if az >= 0 else -az
            m = ax if ax > az else az
            if m >= se:
                continue
            d = y - yb
            ad = d if d >= 0 else -d
            mm = m if m > ad else ad
            if mm < s:
                lv = 256
            elif mm < se:
                lv = (se - mm) * 256 // e
            elif d > 0 and trail > 0 and d < s + trail and m < s:
                # Motion trail above the cube, fading with distance.
                lv = (s + trail - d) * TRAIL_LV // trail
            else:
                continue
            i = path[j]
            k = 3 * i
            v = frame[k] + (cr * lv >> 8)
            frame[k] = v if v < 255 else 255
            v = frame[k + 1] + (cg * lv >> 8)
            frame[k + 1] = v if v < 255 else 255
            v = frame[k + 2] + (cb * lv >> 8)
            frame[k + 2] = v if v < 255 else 255
