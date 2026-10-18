"""Output stage: raw effect frame -> what goes on the wire. MicroPython-safe.

This is the reference implementation of the contract's output stage and is
shared by the firmware runtime, the headless renderer and the browser viewer,
so all three cap and dim a frame identically:

    raw RGB frame -> power cap (on the raw frame) -> brightness -> gamma
    -> colour order -> driver

The cap is computed on the unscaled frame, so brightness 255 means "as bright
as the power budget allows" and lower values dim proportionally (review H7).
All per-frame arithmetic is integer, and process() does not allocate.

Low brightness. Whole-tree effects are capped to about 1/5 of full scale, so
at a few percent brightness the wire values are only 0-2. Three things keep
dim colours from collapsing to red:
- every channel is computed in 8.8 fixed point and rounded to nearest, not
  floored;
- with dither=True (the default), the fraction left over in each channel is
  carried to the next frame (temporal error diffusion), so a target of 0.6
  averages 0.6 over a few frames. Each pixel starts at a different phase, so
  the tree as a whole doesn't flicker in step. A channel that rounds exactly
  never changes, and each channel is at most 1 above its exact target. When
  that could take a frame over the budget (only near the cap, at high
  brightness), the excess is trimmed off, so the cap holds on every frame;
- optional gamma and per-channel white balance.

On MicroPython the per-frame loops run as viper code compiled at import
(it is kept as source because mpy-cross can't emit native code without
-march); everywhere else, and if that fails, the plain-Python versions run.
Both produce identical bytes.
"""

from array import array

# The cap is a current limit. The strip is 12 V WS2811 bullets on a single
# feed, which binds at about 1.5 A; the runtime converts amps to the
# channel-sum budget Output takes.
MA_PER_CHANNEL = 8   # 12 V WS2811 bullet: one channel at 255. Today's 20 is a WS2812/5 V figure.
LIMIT_A = 1.5        # default cap, amps at 12 V

# The original firmware's cap (13% of full white), kept for compatibility.
BUDGET_PERCENT = 13

# Fixed-point precision of the combined cap x brightness scale.
_Q = 12
# Gamma table steps per output level (input resolution 1/_GSTEPS).
_GSTEPS = 8
_GSIZE = 255 * _GSTEPS + 1


def _lut_py(lut, s, gtab, use_gamma):
    """lut[v] = output level for input v at combined scale s (1/65536), in
    8.8 fixed point (0..65280)."""
    if use_gamma:
        for v in range(256):
            lut[v] = gtab[(v * s) >> 13]
    else:
        for v in range(256):
            lut[v] = (v * s + 128) >> 8


def _apply_py(src, dst, res, l0, l1, l2, n3, o0, o1, o2, dither):
    if dither:
        for k in range(0, n3, 3):
            a = l0[src[k + o0]] + res[k]
            dst[k] = a >> 8
            res[k] = a & 255
            a = l1[src[k + o1]] + res[k + 1]
            dst[k + 1] = a >> 8
            res[k + 1] = a & 255
            a = l2[src[k + o2]] + res[k + 2]
            dst[k + 2] = a >> 8
            res[k + 2] = a & 255
    else:
        for k in range(0, n3, 3):
            dst[k] = (l0[src[k + o0]] + 128) >> 8
            dst[k + 1] = (l1[src[k + o1]] + 128) >> 8
            dst[k + 2] = (l2[src[k + o2]] + 128) >> 8


def _trim_py(dst, n3, excess, k):
    """Take excess units off lit channels, one each, starting at k (rotating
    so no LED is always the one dimmed). Returns where to start next time."""
    while excess > 0:
        if dst[k]:
            dst[k] -= 1
            excess -= 1
        k += 1
        if k >= n3:
            k = 0
    return k


_VIPER = """
@micropython.viper
def _lut_v(lut: ptr16, s: int, gtab: ptr16, use_gamma: int):
    v = 0
    while v < 256:
        if use_gamma:
            lut[v] = gtab[(v * s) >> 13]
        else:
            lut[v] = (v * s + 128) >> 8
        v += 1


@micropython.viper
def _apply_v(src: ptr8, dst: ptr8, res: ptr8, l0: ptr16, l1: ptr16, l2: ptr16,
             n3: int, o0: int, o1: int, o2: int, dither: int):
    k = 0
    if dither:
        while k < n3:
            a = l0[src[k + o0]] + res[k]
            dst[k] = a >> 8
            res[k] = a & 255
            a = l1[src[k + o1]] + res[k + 1]
            dst[k + 1] = a >> 8
            res[k + 1] = a & 255
            a = l2[src[k + o2]] + res[k + 2]
            dst[k + 2] = a >> 8
            res[k + 2] = a & 255
            k += 3
    else:
        while k < n3:
            dst[k] = (l0[src[k + o0]] + 128) >> 8
            dst[k + 1] = (l1[src[k + o1]] + 128) >> 8
            dst[k + 2] = (l2[src[k + o2]] + 128) >> 8
            k += 3


@micropython.viper
def _trim_v(dst: ptr8, n3: int, excess: int, k: int) -> int:
    while excess > 0:
        if dst[k]:
            dst[k] = dst[k] - 1
            excess -= 1
        k += 1
        if k >= n3:
            k = 0
    return k
"""

_lut = _lut_py
_apply = _apply_py
_trim = _trim_py
NATIVE = False
try:
    import micropython  # noqa: F401  (MicroPython only)

    _ns = {"micropython": micropython}
    exec(_VIPER, _ns)
    _lut = _ns["_lut_v"]
    _apply = _ns["_apply_v"]
    _trim = _ns["_trim_v"]
    NATIVE = True
    _ns = None
except Exception:
    pass


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
    driver needs. budget is a channel sum; the default is
    budget_for_current() (LIMIT_A at MA_PER_CHANNEL).
    - gamma: None (linear) or an exponent >= 1, applied after the cap and
      brightness, so it never raises a channel above its linear value.
    - dither: carry each channel's sub-LSB remainder to the next frame
      (default True). Calibration frames should use dither=False.
    - white: optional per-channel balance (r, g, b), each 0..1, applied to
      the source channels with brightness (e.g. (1.0, 0.85, 0.7)).

    After each process() call:
    - raw_total: channel sum of the raw frame
    - capped: True when the power cap scaled the frame down
    - factor: cap factor in 1/4096 units (4096 = not capped)
    """

    def __init__(self, n, budget=None, gamma=None, order="RGB", dither=True, white=None):
        self.n = n
        self.budget = budget_for_current() if budget is None else int(budget)
        self.order = bytes(["RGB".index(c) for c in order])
        self.dither = 1 if dither else 0
        self.luts = [array("H", bytes(512))]
        self.white = None
        if white is not None:
            w = [min(4096, max(0, int(c * 4096 + 0.5))) for c in white]
            if w != [4096, 4096, 4096]:
                self.white = w
                self.luts.append(array("H", bytes(512)))
                self.luts.append(array("H", bytes(512)))
        self.gamma = None
        self.gtab = self.luts[0]  # placeholder pointer when gamma is off
        if gamma is not None and gamma != 1.0:
            g = array("H", bytes(2 * _GSIZE))
            top = 255 * 256
            for i in range(_GSIZE):
                g[i] = int(top * (i / (_GSIZE - 1)) ** gamma + 0.5)
            self.gamma = gamma
            self.gtab = g
        # Per-channel remainder, 1/256 LSB. Each starts at a different
        # (hashed) phase so equal targets don't toggle in step.
        self.res = bytearray(3 * n)
        for k in range(3 * n):
            self.res[k] = ((k * 2654435761) >> 16) & 255
        self.trim_at = 0
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
        # Combined cap x brightness scale in 1/65536, rounded: at low
        # brightness a coarser scale would bias every channel low.
        s = (f * brightness * 16 + 127) // 255
        luts = self.luts
        use_gamma = 0 if self.gamma is None else 1
        w = self.white
        if w is None:
            l0 = luts[0]
            _lut(l0, s, self.gtab, use_gamma)
            l1 = l0
            l2 = l0
        else:
            # One table per source channel: luts[c] serves source channel c.
            _lut(luts[0], (s * w[0] + 2048) >> _Q, self.gtab, use_gamma)
            _lut(luts[1], (s * w[1] + 2048) >> _Q, self.gtab, use_gamma)
            _lut(luts[2], (s * w[2] + 2048) >> _Q, self.gtab, use_gamma)
            o = self.order
            l0 = luts[o[0]]
            l1 = luts[o[1]]
            l2 = luts[o[2]]
        o = self.order
        n3 = 3 * self.n
        _apply(src, dst, self.res, l0, l1, l2, n3, o[0], o[1], o[2], self.dither)
        # Rounding and dithering add under 1 per channel to the exact
        # target, about total * s / 65536 (an upper bound here, kept within
        # small ints). If that could cross the budget, enforce it exactly.
        if (((total >> 4) + 1) * ((s >> 4) + 1) >> 8) + n3 > self.budget:
            out = sum(dst)
            if out > self.budget:
                self.trim_at = _trim(dst, n3, out - self.budget, self.trim_at)
