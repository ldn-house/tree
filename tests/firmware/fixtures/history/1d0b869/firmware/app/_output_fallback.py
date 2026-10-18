"""Output stage: raw effect frame -> what goes on the wire. MicroPython-safe.

This is the reference implementation of the contract's output stage and is
shared by the firmware runtime, the headless renderer and the browser viewer,
so all three cap and dim a frame identically:

    raw RGB frame -> power cap (on the raw frame) -> brightness -> gamma
    -> colour order -> driver

The cap is computed on the unscaled frame, so brightness 255 means "as bright
as the power budget allows" and lower values dim proportionally (review H7).
All per-frame arithmetic is integer, and process() does not allocate.
"""

# Fraction of theoretical maximum (every channel at 255) the PSU may supply.
# Matches the original firmware's MAX_POWER (13%).
BUDGET_PERCENT = 13

# Current estimate for a 12V WS2811 pixel: ~20 mA per channel at 255
# (60 mA full white, as the original firmware assumed).
MA_PER_CHANNEL = 20

# Fixed-point precision of the combined cap x brightness scale.
_Q = 12


def budget_for(n, percent=BUDGET_PERCENT):
    return n * 765 * percent // 100


def current_ma(channel_sum):
    """Estimated LED current for a frame whose channels sum to channel_sum."""
    return channel_sum * MA_PER_CHANNEL // 255


class Output:
    """Applies the output stage to a raw frame.

    order is the wire colour order as a string of "R", "G", "B" (e.g. "GRB").
    The renderer and viewer use "RGB"; the firmware passes whatever its
    driver needs. gamma=None keeps the response linear.

    After each process() call:
    - raw_total: channel sum of the raw frame
    - capped: True when the power cap scaled the frame down
    - factor: cap factor in 1/4096 units (4096 = not capped)
    """

    def __init__(self, n, budget=None, gamma=None, order="RGB"):
        self.n = n
        self.budget = budget_for(n) if budget is None else int(budget)
        self.order = bytes(["RGB".index(c) for c in order])
        self.lut = bytearray(256)
        self.gamma = None
        if gamma is not None and gamma != 1.0:
            g = bytearray(256)
            for v in range(256):
                g[v] = int(255 * (v / 255) ** gamma + 0.5)
            self.gamma = g
        self.raw_total = 0
        self.capped = False
        self.factor = 1 << _Q

    def process(self, src, dst, brightness):
        """src: raw frame (RGB). dst: output buffer, same length.
        brightness: 0-255."""
        total = sum(src)
        self.raw_total = total
        one = 1 << _Q
        if total > self.budget:
            f = (self.budget << _Q) // total
            self.capped = True
        else:
            f = one
            self.capped = False
        self.factor = f
        s = f * brightness // 255
        lut = self.lut
        g = self.gamma
        if g is None:
            for v in range(256):
                lut[v] = (v * s) >> _Q
        else:
            for v in range(256):
                lut[v] = g[(v * s) >> _Q]
        o = self.order
        o0 = o[0]
        o1 = o[1]
        o2 = o[2]
        for k in range(0, 3 * self.n, 3):
            dst[k] = lut[src[k + o0]]
            dst[k + 1] = lut[src[k + o1]]
            dst[k + 2] = lut[src[k + o2]]
