"""Output stage: raw effect frame -> what goes on the wire. MicroPython-safe.

This is the reference implementation of the contract's output stage and is
shared by the firmware runtime, the headless renderer and the browser viewer,
so all three cap and dim a frame identically:

    raw RGB frame -> power cap (on the raw frame) -> brightness (above the
    floor) -> white balance -> gamma -> rounding/dithering -> colour order
    -> driver

The cap is computed on the unscaled frame, so brightness 255 means "as bright
as the power budget allows" and lower values dim proportionally (review H7).
All per-frame arithmetic is integer, and process() does not allocate.

Low brightness. Whole-tree effects are capped to about 1/5 of full scale, so
at a few percent brightness the wire values are only 0-2. Three things keep
dim colours from collapsing to red:
- every channel is computed in 8.8 fixed point and rounded to nearest, not
  floored;
- with dither="spatial" (the default), each LED has a fixed rounding
  threshold, spread evenly along the string by a low-discrepancy sequence
  and shared by its three channels, so a target of 0.6 is sent as 1 on
  about 60% of LEDs and 0 on the rest, and a dim LED never shows a stray
  primary (a warm white LED is dark, red, amber or white-ish). Every LED stays within 1 of its exact
  value, the tree-wide average keeps the hue, and a static frame gives the
  same bytes every time (no flicker). The thresholds cover 0..255 evenly,
  so the average is unbiased (a dead zone for small fractions would shift
  dim hues by up to 20%). dither="temporal" (opt-in) instead carries each
  channel's remainder to the next frame; at the tree's ~40 fps that
  flickers visibly at low levels. dither=False rounds to nearest. When
  rounding could take a frame over the budget (only near the cap, at high
  brightness), the excess is trimmed off, so the cap holds on every frame;
- optional gamma and per-channel white balance.

Brightness floor. Below a few percent a channel can only be 0 or 1, so an
LED can't show the colour at all (warm white at 1% is dark, red, amber or
cold "white" LEDs). The floor (floor=, percent of full brightness, default
FLOOR_PCT) maps brightness linearly onto floor..100% from Home Assistant's
1% (brightness 3; 1 and 2, which HA's slider can't pick, give the floor too)
to 255, so the slider still dims smoothly over its whole range and 0 stays
off. At the default 8% under the 1.5 A cap, warm white (tone 1) is
(4-5, 2-3, 0-1) on every LED. Calibration frames use floor=0 (and
brightness 255, which the floor never changes).

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

# Default brightness floor, percent of full brightness (see the docstring).
FLOOR_PCT = 8
# The brightness that lands on the floor: Home Assistant's 1%.
_FLOOR_AT = 3

# Fixed-point precision of the combined cap x brightness scale.
_Q = 12
# Spatial dithering: additive-recurrence step in 1/65536 (golden ratio).
_ALPHA = 40503
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


def _apply_py(src, dst, res, l0, l1, l2, n3, o0, o1, o2, temporal):
    """temporal: res holds remainders carried between frames (updated).
    Otherwise res holds fixed per-channel rounding thresholds (128 everywhere
    is round-to-nearest; varied per LED is spatial dithering) and is only
    read, so a static frame gives identical bytes every time."""
    if temporal:
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
            dst[k] = (l0[src[k + o0]] + res[k]) >> 8
            dst[k + 1] = (l1[src[k + o1]] + res[k + 1]) >> 8
            dst[k + 2] = (l2[src[k + o2]] + res[k + 2]) >> 8


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
             n3: int, o0: int, o1: int, o2: int, temporal: int):
    k = 0
    if temporal:
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
            dst[k] = (l0[src[k + o0]] + res[k]) >> 8
            dst[k + 1] = (l1[src[k + o1]] + res[k + 1]) >> 8
            dst[k + 2] = (l2[src[k + o2]] + res[k + 2]) >> 8
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
    - dither: "spatial" (default; True means the same), "temporal" (carry
      remainders across frames; flickers at low levels) or False/None
      (round to nearest). Calibration frames should use False.
    - white: optional per-channel balance (r, g, b), each 0..1, applied to
      the source channels with brightness (e.g. (1.0, 0.85, 0.7)).
    - floor: brightness floor, percent of full (0 = none); brightness
      3..255 (HA's 1..100%) maps linearly onto floor..100%, 1 and 2 give
      the floor, 0 stays dark. set_floor() changes it.

    After each process() call:
    - raw_total: channel sum of the raw frame
    - capped: True when the power cap scaled the frame down
    - factor: cap factor in 1/4096 units (4096 = not capped)
    """

    def __init__(self, n, budget=None, gamma=None, order="RGB", dither="spatial", white=None,
                 floor=FLOOR_PCT):
        self.n = n
        self.set_floor(floor)
        self.budget = budget_for_current() if budget is None else int(budget)
        self.order = bytes(["RGB".index(c) for c in order])
        if dither is True:
            dither = "spatial"
        if not dither:
            dither = None
        if dither not in (None, "spatial", "temporal"):
            raise ValueError("dither must be 'spatial', 'temporal' or False")
        self.dither = dither
        self.temporal = 1 if dither == "temporal" else 0
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
        # Per channel: temporal remainders (1/256 LSB, updated each frame)
        # or fixed rounding thresholds (read only).
        self.res = bytearray(3 * n)
        if dither == "temporal":
            # Each channel starts at a different (hashed) phase.
            for k in range(3 * n):
                self.res[k] = ((k * 2654435761) >> 16) & 255
        elif dither == "spatial":
            # One threshold per LED, shared by its three channels, from an
            # additive recurrence (fractional parts of i * golden ratio):
            # evenly spread along any run of the string, neighbours far
            # apart. Sharing it means a dim LED's channels round up in
            # nested order, so a warm target only ever shows dark, red,
            # amber or white-ish LEDs, never stray greens or blues.
            # Thresholds use all of 0..255 evenly, so a fraction f rounds up
            # on f/256 of the LEDs: unbiased, which keeps the hue at 1-2%.
            for i in range(n):
                t = ((i * _ALPHA) & 0xFFFF) >> 8
                self.res[3 * i] = t
                self.res[3 * i + 1] = t
                self.res[3 * i + 2] = t
        else:
            for k in range(3 * n):
                self.res[k] = 128
        self.raw_total = 0
        self.capped = False
        self.factor = 1 << _Q

    def set_floor(self, floor):
        """floor: percent of full brightness, 0..100."""
        floor = min(100, max(0, floor))
        self.floor = floor
        # In 1/256 brightness units (65280 = 255).
        self._fq = int(floor * 652.8 + 0.5)

    def level(self, brightness):
        """Effective brightness after the floor, in 1/256 units (0..65280)."""
        if brightness <= 0:
            return 0
        if brightness >= 255:
            return 65280
        fq = self._fq
        if fq == 0:
            return brightness << 8
        if brightness <= _FLOOR_AT:
            return fq
        return fq + ((brightness - _FLOOR_AT) * (65280 - fq) + 126) // (255 - _FLOOR_AT)

    def process(self, src, dst, brightness):
        """src: raw frame (RGB). dst: output buffer, same length.
        brightness: 0-255 (0 is dark; 1..255 map onto floor..100%)."""
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
        # brightness a coarser scale would bias every channel low. With no
        # floor this is exactly (f * brightness * 16 + 127) // 255.
        s = (f * self.level(brightness) + 2032) // 4080
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
        _apply(src, dst, self.res, l0, l1, l2, n3, o[0], o[1], o[2], self.temporal)
        # Rounding and dithering add under 1 per channel to the exact
        # target, about total * s / 65536 (an upper bound here, kept within
        # small ints). If that could cross the budget, enforce it exactly.
        if (((total >> 4) + 1) * ((s >> 4) + 1) >> 8) + n3 > self.budget:
            out = sum(dst)
            if out > self.budget:
                # Always from the same place, so a static frame stays static.
                _trim(dst, n3, out - self.budget, 0)
