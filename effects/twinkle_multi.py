"""Multicolour fairy lights that gently pulse, occasionally changing colour."""

from array import array

from effects._util import SIN

NAME = "twinkle_multi"
DURATION_MS = 10000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
}
NEEDS_COORDS = False

# Red, green, blue, warm yellow.
COLORS = bytes((255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 180, 0))
# Pulse rate per LED, in 1/256ths of a 16-bit phase per ms: 728-2427 is one
# pulse every 6.9-23 s, the original 0.03-0.10 rad per ~110 ms frame.
SPEED_MIN = 728
SPEED_SPAN = 1699
CHANGE_CHANCE = 77  # out of 256: ~30% chance to change colour per pulse


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.params = ctx.params
        rand = ctx.random.getrandbits
        self.rand = rand
        self.color = bytearray(n)
        self.phase = array("H", bytes(2 * n))
        self.rate = array("H", bytes(2 * n))
        for i in range(n):
            self.color[i] = rand(2)
            self.phase[i] = rand(16)
            self.rate[i] = SPEED_MIN + (rand(10) * SPEED_SPAN >> 10)
        self.last = 0
        self.raw = None
        self.speed = 256

    def render(self, t_ms, frame):
        s = self.params.get("speed", 1.0)
        if s is not self.raw:
            self.raw = s
            self.speed = int(s * 256)
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0
        dt = dt * self.speed >> 8
        rand = self.rand
        color = self.color
        phase = self.phase
        rate = self.rate
        for i in range(self.n):
            p = phase[i] + (rate[i] * dt >> 8)
            if p > 65535:
                p &= 65535
                if rand(8) < CHANGE_CHANCE:
                    color[i] = rand(2)
                    rate[i] = SPEED_MIN + (rand(10) * SPEED_SPAN >> 10)
            phase[i] = p
            # 60-100% brightness
            b = 153 + (102 * SIN[p >> 8] >> 8)
            c = color[i] * 3
            k = 3 * i
            frame[k] = COLORS[c] * b >> 8
            frame[k + 1] = COLORS[c + 1] * b >> 8
            frame[k + 2] = COLORS[c + 2] * b >> 8
