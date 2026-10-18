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

# The cap is a current limit. The strip is 12 V WS2811 bullets on a single
# feed, which binds at about 1.5 A; the runtime converts amps to the
# channel-sum budget Output takes.
MA_PER_CHANNEL = 8   # 12 V WS2811 bullet: one channel at 255. Today's 20 is a WS2812/5 V figure.
LIMIT_A = 1.5        # default cap, amps at 12 V

# The original firmware's cap (13% of full white), kept for compatibility.
BUDGET_PERCENT = 13

# Fixed-point precision of the combined cap x brightness scale.
_Q = 12


def budget_for_current(limit_a=LIMIT_A, ma_per_channel=MA_PER_CHANNEL):
    """Channel-sum budget for a current limit (what Output(budget=...) takes)."""
    return int(limit_a * 1000 * 255) // ma_per_channel


def current_a(channel_sum, ma_per_channel=MA_PER_CHANNEL):
    return channel_sum * ma_per_channel / 255000


def budget_for(n, percent=BUDGET_PERCENT):
    return n * 765 * percent // 100


def current_ma(channel_sum):
    """Estimated LED current in mA for a frame whose channels sum to channel_sum."""
    return channel_sum * MA_PER_CHANNEL // 255


class Output:
    """Applies the output stage to a raw frame.

    order is the wire colour order as a string of "R", "G", "B" (e.g. "GRB").
    The renderer and viewer use "RGB"; the firmware passes whatever its
    driver needs. gamma=None keeps the response linear. budget is a channel
    sum; the default is budget_for_current() (LIMIT_A at MA_PER_CHANNEL).

    After each process() call:
    - raw_total: channel sum of the raw frame
    - capped: True when the power cap scaled the frame down
    - factor: cap factor in 1/4096 units (4096 = not capped)
    """

    def __init__(self, n, budget=None, gamma=None, order="RGB"):
        self.n = n
        self.budget = budget_for_current() if budget is None else int(budget)
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
