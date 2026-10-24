"""F1 winner: the winning team's colour, celebrating.

The tree glows in the team colour while bright rings of it rise from the
bottom to the star one after another, and a few LEDs at a time catch a
soft white glint that fades over half a second, like camera flashes in the
crowd. The glints are sparse (a handful at once) and slow to fade, so
nothing strobes.

Home Assistant's F1 automation sets the colour from F1 Sensor's
sensor.f1_top_three_p1 team_color_rgb.
"""

from array import array

from effects._output import budget_for_current
from effects._util import ONE, ColorParam, Layout, Param, Phase, rgb_spec

NAME = "f1_winner"
DURATION_MS = 20000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "color": rgb_spec(255, 128, 0),  # until Home Assistant says who won
}
NEEDS_COORDS = True
AUTO = False

RING_MS = 1100  # a new ring leaves the bottom this often at speed 1
RISE_MS = 2200  # and takes this long to reach the top
RING_W = 0.09  # ring half-height, tree heights
BASE_SHARE = 0.35  # glow's share of the budget, full tree
RING_SHARE = 3.0  # per-LED sum on a ring, as a multiple of the average budget
GLINTS_PER_S = 7  # new glints a second, on average
GLINT_MS = 500  # a glint fades over this


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h
        self.rnd = ctx.random
        count = lay.count or n
        per_led = budget_for_current() / count
        self.base_sum = int(BASE_SHARE * per_led)
        self.ring_sum = int(RING_SHARE * per_led)
        self.hw = int(RING_W * ONE)
        self.glint = bytearray(n)  # 0..255, decaying
        self.speed = Param(ctx, "speed", 1.0)
        self.rings = Phase(RING_MS)
        self.color = ColorParam(ctx, "color", PARAMS["color"][0])
        self.base = 0  # /256 colour scale for the glow
        self.peak = 0  # and for a ring's centre
        self.last = 0
        self.acc = 0  # glint spawning, in 1/1000 glints
        self.t = 0  # ms at speed 1
        self.sub = 0

    def render(self, t_ms, frame):
        c = self.color
        if c.update():
            s = c.r + c.g + c.b or 1
            self.base = self.base_sum * 256 // s
            self.peak = self.ring_sum * 256 // s
        speed = self.speed.get()
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.sub += dt * speed
        step = self.sub >> 12
        self.sub &= ONE - 1
        self.t += step
        if self.t >= 1 << 28:
            self.t -= RING_MS << 16
        t = self.t
        # Glints: fade the old, spawn new at random.
        glint = self.glint
        fade = step * 255 // GLINT_MS
        n = self.n
        if fade:
            for i in range(n):
                g = glint[i]
                if g:
                    glint[i] = g - fade if g > fade else 0
        self.acc += step * GLINTS_PER_S
        while self.acc >= 1000:
            self.acc -= 1000
            i = self.rnd.getrandbits(16) % n
            if self.known[i]:
                glint[i] = 255
        base = self.base
        peak = self.peak
        cr = c.r
        cg = c.g
        cb = c.b
        hw = self.hw
        rise = RISE_MS
        h = self.h
        known = self.known
        # The rings in flight: their heights this frame.
        k_now = t // RING_MS
        for i in range(n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            lv = base
            hi = h[i]
            j = 0
            while j * RING_MS < rise + RING_MS:
                age = t - (k_now - j) * RING_MS
                j += 1
                if age < 0 or age >= rise:
                    continue
                pos = age * (ONE + 2 * hw) // rise - hw
                d = hi - pos
                if d < 0:
                    d = -d
                if d < hw:
                    q = (hw - d) * 256 // hw
                    v = base + ((peak - base) * (q * q >> 8) >> 8)
                    if v > lv:
                        lv = v
            r = cr * lv >> 8
            g = cg * lv >> 8
            b = cb * lv >> 8
            gl = glint[i]
            if gl:
                # Towards white, at about the ring's brightness.
                w = peak * 255 >> 8
                if w > 255:
                    w = 255
                r += (w - r) * gl >> 8 if w > r else 0
                g += (w - g) * gl >> 8 if w > g else 0
                b += (w - b) * gl >> 8 if w > b else 0
            frame[k] = r if r < 255 else 255
            frame[k + 1] = g if g < 255 else 255
            frame[k + 2] = b if b < 255 else 255
