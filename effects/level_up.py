"""Level up: a new colour climbs the tree one height band per beat.

The tree is split into `steps` height bands. On each beat the next band up
switches to the new colour with a short brightening tick as it lands, so
the fill reads as discrete steps rather than a smooth wipe. When the top
band lands the whole tree finishes with one bright flash that settles into
a glow (finish 0) or one slow glow pulse (finish 1), holds briefly, and
the next level starts in the next colour of the palette.

The colours are the class attribute `palette`, separate from the timing,
so a variant (an F1 race start, say) only needs a subclass:

    class Effect(level_up.Effect):
        palette = bytes((0, 255, 40))  # green, every level
        once = True  # one level, then hold the glow (f1_start)

Power: the steady glow is sized to about a third of the default 1.5 A cap
for a full tree, so the finish flash and pulse (which the cap scales to the
full budget) still read as about three times brighter than the glow.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, MASK, Layout, Param

NAME = "level_up"
DURATION_MS = 30000
PARAMS = {
    "speed": (1.0, 0.25, 4.0),  # beat rate; 1.0 = one band every BEAT_MS
    "steps": (8, 3, 16),  # height bands
    "finish": (0, 0, 1),  # 0 = one flash settling to a glow, 1 = one slow glow pulse
}
NEEDS_COORDS = True

MAX_STEPS = 16
BEAT_MS = 550  # one band per beat at speed 1
LEAD_MS = 450  # the old colour dims before the first band lands
TICK_MS = 380  # a landed band's brightening decays over this
FLASH_MS = 1100  # flash: peak, then settle to the glow
PULSE_MS = 2200  # pulse: glow -> peak -> glow
HOLD_MS = 1300  # glow before the next level

# Palette, one RGB triple per level, cycling. Alternates warm and cool.
PALETTE = bytes((
    20, 255, 40,  # green
    255, 150, 0,  # gold
    0, 120, 255,  # blue
    255, 30, 90,  # rose
    0, 230, 200,  # teal
    255, 70, 0,  # orange
    140, 40, 255,  # violet
))

GLOW_SHARE = 0.33  # full-tree glow as a share of the default power budget
PEAK_SHARE = 1.1  # flash and pulse peak: just over the budget, so the cap holds it at the limit
UNDER = 3  # the previous colour sits at glow >> UNDER while the next fills
FLASH_WHITE = 100  # flash peak blends the colour this far (/256) towards white

# Level phases.
_FILL = 0
_FINISH = 1
_HOLD = 2


class Effect:
    palette = PALETTE
    once = False

    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h
        count = lay.count or n
        # Sorted heights of the known LEDs, for the band boundaries.
        hs = sorted(lay.h[i] for i in range(n) if lay.known[i])
        self.sorted_h = array("H", hs or [0])
        hs = None
        # Per-LED channel sums that put a full tree at GLOW_SHARE and
        # PEAK_SHARE of the default budget.
        budget = budget_for_current()
        self.glow_sum = int(GLOW_SHARE * budget / count)
        self.peak_sum = int(PEAK_SHARE * budget / count)
        self.band = bytearray(n)
        self.bounds = array("H", bytes(2 * (MAX_STEPS + 1)))
        self.bandcol = bytearray(3 * MAX_STEPS)
        self.steps_n = 0
        self.speed = Param(ctx, "speed", 1.0)
        self.steps = Param(ctx, "steps", 8, 1)
        self.finish = Param(ctx, "finish", 0, 1)
        self.last = 0
        self.sub = 0
        self.t = 0  # ms into the current level, at speed 1
        self.level = 0

    def set_steps(self, steps):
        # Boundaries halfway between equal heights and equal LED counts:
        # the tree is denser low down, so equal heights would make the lower
        # bands crowded and equal counts would make the top band most of
        # the tree.
        self.steps_n = steps
        s = self.sorted_h
        m = len(s)
        b = self.bounds
        for k in range(steps + 1):
            b[k] = (k * ONE // steps + s[k * (m - 1) // steps]) >> 1
        b[0] = 0
        b[steps] = ONE + 1
        band = self.band
        h = self.h
        for i in range(self.n):
            k = 0
            while k < steps - 1 and h[i] >= b[k + 1]:
                k += 1
            band[i] = k

    def glow(self, o):
        p = self.palette
        s = p[o] + p[o + 1] + p[o + 2]
        g = self.glow_sum * 256 // s if s else 256
        return g if g < 256 else 256

    def render(self, t_ms, frame):
        steps = self.steps.get()
        steps = 3 if steps < 3 else MAX_STEPS if steps > MAX_STEPS else steps
        if steps != self.steps_n:
            self.set_steps(steps)

        # Level clock at speed 1, in whole ms.
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.sub += dt * self.speed.get()
        self.t += self.sub >> 12
        self.sub &= MASK

        finish = 1 if self.finish.get() else 0
        fill_ms = LEAD_MS + steps * BEAT_MS
        finish_ms = PULSE_MS if finish else FLASH_MS
        if self.once and self.t > fill_ms + finish_ms:
            self.t = fill_ms + finish_ms  # hold the first level's glow
        elif self.t >= fill_ms + finish_ms + HOLD_MS:
            self.t = 0
            self.level += 1
            if self.level >= 1 << 20:
                self.level = 1
        t = self.t

        pal = self.palette
        ncol = len(pal) // 3
        o = (self.level % ncol) * 3
        r = pal[o]
        g = pal[o + 1]
        b = pal[o + 2]
        glow = self.glow(o)
        # The previous level's colour, dimmed (nothing before the first).
        po = ((self.level - 1) % ncol) * 3
        if self.level:
            pg = self.glow(po)
            if t < LEAD_MS:
                # Fade from its glow down to the under level.
                pg = pg - (pg - (pg >> UNDER)) * t // LEAD_MS
            else:
                pg >>= UNDER
        else:
            pg = 0
        pr = pal[po] * pg >> 8
        pgg = pal[po + 1] * pg >> 8
        pb = pal[po + 2] * pg >> 8

        bc = self.bandcol
        if t < fill_ms:
            for k in range(steps):
                j = 3 * k
                age = t - LEAD_MS - k * BEAT_MS
                if age < 0:
                    bc[j] = pr
                    bc[j + 1] = pgg
                    bc[j + 2] = pb
                    continue
                lv = glow
                if age < TICK_MS:
                    # Lands at full colour, eases back to the glow.
                    q = (TICK_MS - age) * 256 // TICK_MS
                    lv += (256 - glow) * (q * q >> 8) >> 8
                bc[j] = r * lv >> 8
                bc[j + 1] = g * lv >> 8
                bc[j + 2] = b * lv >> 8
        else:
            ft = t - fill_ms
            wr = r
            wg = g
            wb = b
            if ft >= finish_ms:
                q = 0
            elif finish:
                # One slow pulse: a smooth bump, peaking halfway.
                x = ft * 512 // finish_ms - 256  # -256..256
                q = 256 - (x * x >> 8)  # 0..256..0
                q = q * q >> 8  # softer shoulders
            else:
                # Fast attack, slow quadratic settle, whitened at the peak.
                attack = finish_ms >> 4
                if ft < attack:
                    q = 256 * ft // attack
                else:
                    q = (finish_ms - ft) * 256 // (finish_ms - attack)
                    q = q * q >> 8
                w = FLASH_WHITE * q >> 8
                wr = r + ((255 - r) * w >> 8)
                wg = g + ((255 - g) * w >> 8)
                wb = b + ((255 - b) * w >> 8)
            # Interpolate the full tree's channel sum from the glow to the
            # peak, so the peak only just reaches the cap and the settle is
            # visible from its first frame instead of hiding under the cap.
            s = wr + wg + wb or 1
            lv = (self.glow_sum + ((self.peak_sum - self.glow_sum) * q >> 8)) * 256 // s
            if lv > 256:
                lv = 256
            cr = wr * lv >> 8
            cg = wg * lv >> 8
            cb = wb * lv >> 8
            for k in range(steps):
                j = 3 * k
                bc[j] = cr
                bc[j + 1] = cg
                bc[j + 2] = cb

        known = self.known
        band = self.band
        for i in range(self.n):
            k = 3 * i
            if known[i]:
                j = 3 * band[i]
                frame[k] = bc[j]
                frame[k + 1] = bc[j + 1]
                frame[k + 2] = bc[j + 2]
            else:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
