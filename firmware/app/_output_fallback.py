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
at a few percent brightness the wire values are only a few levels. On the
tree, a one-level difference in a small channel is visible as a different
tint on that LED (blue 0 next to blue 1 reads as yellow next to white), so:
- every channel is computed in 8.8 fixed point and rounded, not floored;
- a channel whose target is below dither_min levels (default DITHER_MIN = 9)
  is rounded to nearest on every LED, so a solid colour is one colour on
  the whole tree. Nearest is the closest integer to each channel's target,
  which also keeps the hue closest: warm white (255, 87, 14) at the floor is
  (3, 1, 0) on every LED;
- at and above dither_min, where one level is at most about 1/9 of the
  channel and wasn't visible on the tree, dither="spatial" (the default)
  gives each LED a fixed rounding threshold, spread evenly along the string
  by a low-discrepancy sequence and shared by its three channels: a target
  of 9.6 is sent as 10 on about 60% of LEDs and 9 on the rest. Every LED
  stays within 1 of its exact value, the tree-wide average matches, and a
  static frame gives the same bytes every time (no flicker).
  dither="temporal" (opt-in) instead carries each channel's remainder to
  the next frame, which flickers at the tree's ~40 fps. dither=False rounds
  to nearest everywhere. When rounding could take a frame over the budget
  (only near the cap, at high brightness), the excess is trimmed off, so
  the cap holds on every frame;
- optional gamma and per-channel white balance.

Brightness floor. Below a few levels no rounding can show a colour (warm
white's red would be 1 or 2). The floor (floor=, percent of full
brightness, default FLOOR_PCT = 4) maps brightness linearly onto
floor..100% from Home Assistant's 1% (brightness 3; 1 and 2, which HA's
slider can't pick, give the floor too) to 255, so 0 stays off and the
slider covers the whole usable range. 4% is the lowest level at which warm
white (tone 2) is still its own hue, (3, 1, 0); at 3% it would be (2, 1, 0),
too yellow. Calibration frames use floor=0 (and brightness 255, which the
floor never changes).

Colour floor. Whether a few levels still show a colour depends on the
colour: a pure red or green is itself at any level, but warm white
(255, 87, 14) at red 3-8 has no blue and reads as yellow, and only from
about red 11, (11, 4, 1), does it look warm white again. ColourFloor finds,
for the frame an effect is showing, the lowest level from which every step
up the slider keeps the colours within a threshold (accuracy=, default
COLOUR_ACCURACY = 0.01), and the stage uses the higher of that and the
user's floor (set_colour_floor(); floor_pct() is the result):
- drift is the distance in CIE 1976 u'v' between a pixel's wire colour and
  its exact target, treating wire values as linear light (the WS2811 dims
  by PWM) with sRGB primaries. On the tree, (3, 1, 0) for warm white is
  0.012 and looked yellow, (22, 8, 1) is 0.005 and looked warm white; every
  blue-less step of warm white is about 0.012-0.016, so 0.01 means "never
  the yellow". Rounding to all-dark isn't drift (the user's floor is for
  that), so pure single-channel colours never hold the floor up;
- for colours further from white than the warm whites (COLOUR_HUE_FROM,
  just past warm_white tone 2's 0.104), the allowed drift grows with the
  distance, so it is a hue angle rather than a distance (about 5.5 degrees
  around white at 0.01): the same u'v' step is a much smaller change of
  hue for a saturated colour, and a fixed distance held rainbow_cycle at
  35% on the tree, visibly too bright for Home Assistant's 1%. warm_white
  tones 0-2 (0.065-0.104) keep the fixed threshold, and the more orange
  tone 3 (0.122) gets 16% more. On the tree, warm_white tone 0
  (255, 140, 50) at (5, 3, 1) (0.006) looked right, at (4, 2, 1) and
  (6, 4, 1) (0.011, 0.015) plain white, at (3, 1, 1) (0.045) pink;
- the wire colour is what this stage sends: channels below dither_min are
  rounded to a whole level, and dithered channels count at their exact
  value (dithering is unbiased, and its one-level splits there weren't
  visible);
- levels are checked in half-level steps of each pixel's brightest channel,
  up to COLOUR_FLOOR_MAX (50%): a drift higher up can't be fixed by a floor,
  and the floor never goes above that;
- in a mixed frame, pixels dimmer than 1/COLOUR_DIM of the frame's
  brightest are skipped (fades and tails, which would otherwise need many
  times the level), up to COLOUR_SAMPLES of the rest are analysed, evenly
  spread along the string, and the floor is the level that COLOUR_PCT
  (90%) of them need;
- an animated effect changes colour, so frames are analysed at
  COLOUR_AT_MS (0, 1, 3 and 6 s) after the effect, its params, the
  coordinates or the threshold change, pooling the pixels; each sample
  sets the floor when it is done, and until the first one is, the previous
  floor stays. On the device a sample is spread over the following frames,
  at most a few milliseconds each (step(us=)): one colour took about 10 ms
  on the Pico, so a rainbow analysed all at once held a frame for 320 ms.
  Nothing runs per frame otherwise.

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
FLOOR_PCT = 4
# Channel targets below this many levels are rounded to nearest on every LED
# rather than dithered (see the docstring).
DITHER_MIN = 9
# The brightness that lands on the floor: Home Assistant's 1%.
_FLOOR_AT = 3

# Colour floor (see the docstring). Accuracy is the u'v' drift allowed at
# every level above the floor; 0 turns the colour floor off, and other
# settings are kept within COLOUR_MIN..COLOUR_MAX.
COLOUR_ACCURACY = 0.01
COLOUR_MIN = 0.005
COLOUR_MAX = 0.05
COLOUR_FLOOR_MAX = 50  # percent of full brightness
COLOUR_PCT = 90  # share of the analysed pixels kept within the threshold
COLOUR_SAMPLES = 32  # pixels analysed per frame, at most
COLOUR_DIM = 2  # pixels below 1/COLOUR_DIM of the frame's brightest are skipped
COLOUR_AT_MS = (0, 1000, 3000, 6000)  # when to analyse, after a change
# u'v' of the white point (D65) and the distance from it beyond which the
# allowed drift grows with the distance (a hue angle).
COLOUR_WHITE = (0.1978, 0.4683)
COLOUR_HUE_FROM = 0.105

# CIE 1976 u'v' from linear sRGB (X, Y, Z = M . rgb):
# u' = 4X / D, v' = 9Y / D, D = X + 15Y + 3Z.
_M = ((0.4124, 0.3576, 0.1805), (0.2126, 0.7152, 0.0722), (0.0193, 0.1192, 0.9505))
_U = tuple(4 * _M[0][c] for c in range(3))
_V = tuple(9 * _M[1][c] for c in range(3))
_D = tuple(_M[0][c] + 15 * _M[1][c] + 3 * _M[2][c] for c in range(3))
# The same in integers, for ColourFloor (floats allocate on MicroPython):
# coefficients x 32 and channels scaled to at most 8191 (u'v' doesn't
# depend on scale) give u', v' in 1/32768 with every intermediate, and the
# squared distance, below 2^30 (a small int). The rounded coefficients put
# the distance within about 5% of the exact one, so ColourFloor decides
# within 6% of the threshold in floats.
_KU0, _KU1, _KU2 = (int(c * 32 + 0.5) for c in _U)
_KV0, _KV1, _KV2 = (int(c * 32 + 0.5) for c in _V)
_KD0, _KD1, _KD2 = (int(c * 32 + 0.5) for c in _D)
_UVQ = 32768

# Fixed-point precision of the combined cap x brightness scale.
_Q = 12
# Spatial dithering: additive-recurrence step in 1/65536 (golden ratio).
_ALPHA = 40503
# Gamma table steps per output level (input resolution 1/_GSTEPS).
_GSTEPS = 8
_GSIZE = 255 * _GSTEPS + 1


def _lut_py(lut, s, gtab, use_gamma, umax):
    """lut[v] = output level for input v at combined scale s (1/65536), in
    8.8 fixed point (0..65280). Levels below umax (8.8) are rounded to a
    whole level here, so every LED's rounding threshold (0..255) leaves them
    unchanged: no dithering below dither_min."""
    for v in range(256):
        if use_gamma:
            a = gtab[(v * s) >> 13]
        else:
            a = (v * s + 128) >> 8
        if a < umax:
            a = ((a + 128) >> 8) << 8
        lut[v] = a


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
def _lut_v(lut: ptr16, s: int, gtab: ptr16, use_gamma: int, umax: int):
    v = 0
    while v < 256:
        if use_gamma:
            a = gtab[(v * s) >> 13]
        else:
            a = (v * s + 128) >> 8
        if a < umax:
            a = ((a + 128) >> 8) << 8
        lut[v] = a
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


def colour_accuracy(v):
    """A colour accuracy setting within its bounds: 0 (or less) is off."""
    if not v > 0:
        return 0
    return min(COLOUR_MAX, max(COLOUR_MIN, v))


def uv(rgb):
    """CIE 1976 (u', v') of a linear RGB colour, or None for black."""
    r, g, b = rgb
    d = _D[0] * r + _D[1] * g + _D[2] * b
    if not d:
        return None
    return ((_U[0] * r + _U[1] * g + _U[2] * b) / d, (_V[0] * r + _V[1] * g + _V[2] * b) / d)


def drift(out, target):
    """u'v' distance between a wire colour and its target (both linear).
    All-dark output is 0: it is no colour rather than a wrong one."""
    a = uv(out)
    b = uv(target)
    if a is None or b is None:
        return 0.0
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def tolerance(target, accuracy):
    """The u'v' drift allowed for a target colour (linear): accuracy, or
    for a colour further than COLOUR_HUE_FROM from white, accuracy times
    its distance over COLOUR_HUE_FROM (a hue angle)."""
    p = uv(target)
    if p is None:
        return accuracy
    d = ((p[0] - COLOUR_WHITE[0]) ** 2 + (p[1] - COLOUR_WHITE[1]) ** 2) ** 0.5
    return accuracy * d / COLOUR_HUE_FROM if d > COLOUR_HUE_FROM else accuracy


def _uvq(x0, x1, x2):
    """(u', v') in 1/32768 of a linear colour (non-negative ints), in
    integers. ColourFloor._over() has the same inline."""
    m = max(x0, x1, x2)
    while m > 8191:
        x0 = (x0 + 1) >> 1
        x1 = (x1 + 1) >> 1
        x2 = (x2 + 1) >> 1
        m = (m + 1) >> 1
    d = _KD0 * x0 + _KD1 * x1 + _KD2 * x2
    n = (_KU0 * x0 + _KU1 * x1 + _KU2 * x2) << 7
    q = n // d
    n = (n - q * d) << 6
    r = n // d
    u = (((q << 6) + r) << 2) + (((n - r * d) << 2) // d)
    n = (_KV0 * x0 + _KV1 * x1 + _KV2 * x2) << 7
    q = n // d
    n = (n - q * d) << 6
    r = n // d
    return u, (((q << 6) + r) << 2) + (((n - r * d) << 2) // d)


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
      the floor, 0 stays dark. set_floor() changes it. A ColourFloor can
      raise it (set_colour_floor()); the higher of the two applies, and
      floor_pct() returns it.
    - dither_min: channel targets below this many levels are rounded to
      nearest on every LED instead of dithered (0 dithers everything).

    After each process() call:
    - raw_total: channel sum of the raw frame
    - capped: True when the power cap scaled the frame down
    - factor: cap factor in 1/4096 units (4096 = not capped)
    """

    def __init__(self, n, budget=None, gamma=None, order="RGB", dither="spatial", white=None,
                 floor=FLOOR_PCT, dither_min=DITHER_MIN):
        self.n = n
        self._cq = 0
        self.colour_floor = 0
        self.set_floor(floor)
        self.dither_min = dither_min
        self.umax = min(65280, max(0, int(dither_min * 256)))
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
        self._uq = int(floor * 652.8 + 0.5)
        self._fq = max(self._uq, self._cq)

    def set_colour_floor(self, level):
        """level: the colour floor in 1/256 brightness units (0..65280), as
        ColourFloor finds it; the higher of it and the floor applies."""
        self._cq = min(65280, max(0, int(level)))
        self.colour_floor = round(self._cq / 652.8, 2)
        self._fq = max(self._uq, self._cq)

    def floor_pct(self):
        """The floor in effect, percent: the higher of floor and colour floor."""
        return round(self._fq / 652.8, 2)

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
            _lut(l0, s, self.gtab, use_gamma, self.umax)
            l1 = l0
            l2 = l0
        else:
            # One table per source channel: luts[c] serves source channel c.
            _lut(luts[0], (s * w[0] + 2048) >> _Q, self.gtab, use_gamma, self.umax)
            _lut(luts[1], (s * w[1] + 2048) >> _Q, self.gtab, use_gamma, self.umax)
            _lut(luts[2], (s * w[2] + 2048) >> _Q, self.gtab, use_gamma, self.umax)
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


class ColourFloor:
    """Finds the colour floor for an effect from a few of its raw frames and
    sets it on the stage (see the docstring). Call reset() whenever the
    effect, its params, the coordinates or the accuracy change; then, when
    due(ms since the reset) says so, either add(raw) to analyse a frame at
    once, or start(raw) and step() a little at a time until it returns
    True. Nothing else runs per frame. accuracy: u'v' drift allowed, 0 for
    no colour floor.

    After each analysed frame: pct (the colour floor, percent), level (1/256
    units), pixels (analysed since the reset), us (time spent on the frame,
    where the platform has ticks_us)."""

    def __init__(self, stage, accuracy=COLOUR_ACCURACY):
        self.stage = stage
        self.accuracy = colour_accuracy(accuracy)
        n = stage.n
        self.raw = bytearray(3 * n)
        self.lit = array("H", bytes(2 * n))
        self.needs = []
        self.seen = {}
        self.at = 0
        self.busy = False
        self._i = 0
        self._j = 0
        self._take = 0
        self._count = 0
        self._dim = 0
        self._f = 1 << _Q
        self.level = 0
        self.pct = 0
        self.pixels = 0
        self.us = 0

    def set_accuracy(self, accuracy):
        self.accuracy = colour_accuracy(accuracy)
        self.reset()

    def reset(self):
        """Start over: the next due() is at 0 ms. The stage keeps the current
        colour floor until the next frame is analysed."""
        self.needs = []
        self.at = 0
        self.busy = False

    def due(self, ms):
        return not self.busy and self.at < len(COLOUR_AT_MS) and ms >= COLOUR_AT_MS[self.at]

    def add(self, raw):
        """Analyse a raw frame (before the cap, as the effect rendered it) at
        once. Returns the colour floor in percent."""
        self.start(raw)
        self.step()
        return self.pct

    def start(self, raw):
        """Take a copy of a raw frame to analyse (no per-pixel work)."""
        t0 = _ticks_us()
        self.at += 1
        self.busy = True
        self._i = 0
        self._j = 0
        self._count = 0
        self._take = 0
        self.seen = {}
        st = self.stage
        src = self.raw
        src[:] = raw
        total = sum(src)
        if total:
            self._f = (st.budget << _Q) // total if total > st.budget else 1 << _Q
        if self.accuracy > 0 and total and self._f > 0:
            self._dim = max(src) // COLOUR_DIM
        else:
            self._i = st.n  # nothing to analyse
        self.us = _ticks_diff(_ticks_us(), t0)

    def step(self, pixels=1 << 30, us=0):
        """Go on with the started frame: find its lit pixels, then analyse
        up to `pixels` more of them, stopping early once `us` microseconds
        have passed (0: no limit; each call makes some progress, and one
        colour can't be split). Returns True when the frame is done and the
        colour floor is set."""
        if not self.busy:
            return True
        t0 = _ticks_us()
        n = self.stage.n
        i = self._i
        if i < n:
            src = self.raw
            lit = self.lit
            dim = self._dim
            count = self._count
            while i < n:
                end = min(n, i + 50)
                while i < end:
                    k = 3 * i
                    if src[k] > dim or src[k + 1] > dim or src[k + 2] > dim:
                        lit[count] = i
                        count += 1
                    i += 1
                if us and _ticks_diff(_ticks_us(), t0) >= us:
                    break
            self._i = i
            self._count = count
            if i < n:
                self.us += _ticks_diff(_ticks_us(), t0)
                return False
            self._take = min(count, COLOUR_SAMPLES)
        lmax = COLOUR_FLOOR_MAX * 65280 // 100
        acc = self.accuracy
        src = self.raw
        lit = self.lit
        seen = self.seen
        needs = self.needs
        count = self._count
        take = self._take
        f = self._f
        j = self._j
        end = min(take, j + pixels)
        while j < end:
            k = 3 * lit[j * count // take]
            key = src[k] << 16 | src[k + 1] << 8 | src[k + 2]
            need = seen.get(key)
            j += 1
            if need is None:
                need = self._need(src[k], src[k + 1], src[k + 2], f, lmax, acc)
                seen[key] = need
                needs.append(need)
                if us and _ticks_diff(_ticks_us(), t0) >= us:
                    break
            else:
                needs.append(need)
        self._j = j
        done = j >= take
        if done:
            self.busy = False
            self.seen = {}
            level = 0
            if self.accuracy > 0 and needs:
                needs.sort()
                level = needs[(len(needs) * COLOUR_PCT + 99) // 100 - 1]
            self.pixels = len(needs)
            self.level = level
            self.stage.set_colour_floor(level)
            self.pct = self.stage.colour_floor
        self.us += _ticks_diff(_ticks_us(), t0)
        return done

    def _need(self, r, g, b, f, lmax, acc):
        """Lowest level (1/256 units) from which this colour stays within its
        tolerance() at every level up to lmax.

        The wire colour only changes where a channel crosses a rounding
        boundary (k + 0.5 levels) or dither_min, so the levels split into
        intervals of one rounded colour. Within one, a dithered (exact)
        channel moves the colour along a straight line in u'v' towards the
        fixed target, so the drift is largest at an end: checking both ends
        of every interval, from the top down, finds the highest level that
        drifts. (With gamma the target moves too; a grid of extra checks
        stands in for that.)"""
        if (r > 0) + (g > 0) + (b > 0) < 2:
            return 0  # a single channel never changes colour
        st = self.stage
        self._px = (r, g, b)
        w = st.white
        t = (r, g, b) if w is None else (r * w[0], g * w[1], b * w[2])
        if st.gamma is not None:
            # At full level, as gamma shapes it (the target moves with the
            # level, but its distance from white hardly does).
            t = (t[0] ** st.gamma, t[1] ** st.gamma, t[2] ** st.gamma)
        self._thr = tolerance(t, acc)
        thr2 = self._thr * self._thr
        self._lo2 = int(thr2 * 0.88 * _UVQ * _UVQ)
        self._mid2 = int(thr2 * _UVQ * _UVQ)
        self._hi2 = int(thr2 * 1.13 * _UVQ * _UVQ) + 1
        if st.gamma is None:
            # The target's chromaticity doesn't depend on the level.
            w = st.white
            if w is None:
                self._t = (r, g, b)
                self._tuv = _uvq(r << 5, g << 5, b << 5)
            else:
                self._t = (r * w[0], g * w[1], b * w[2])
                self._tuv = _uvq(r * w[0], g * w[1], b * w[2])
        smax = (f * lmax + 2032) // 4080
        # Scales (s, as process() computes it) at which a channel's output
        # changes: every rounding boundary below dither_min, and dither_min.
        cuts = []
        umax = st.umax
        for c in range(3):
            v = self._px[c]
            if not v:
                continue
            a = 128
            while True:
                if a > umax:
                    a = umax
                sc = self._scale_for(c, v, a)
                if sc >= smax:
                    break
                if sc > 0:
                    cuts.append(sc)
                if a >= umax:
                    break
                a += 256
        if st.gamma is not None:
            step = max(1, smax >> 6)
            cuts.extend(range(step, smax, step))
        cuts.append(smax + 1)
        cuts.sort()
        hi = smax
        i = len(cuts) - 1
        while hi > 0:
            while i >= 0 and cuts[i] > hi:
                i -= 1
            lo = cuts[i] if i >= 0 else 1
            # [lo, hi] has one rounded colour.
            x = self._over(hi)
            if x == 1:
                return lmax if hi >= smax else self._level(hi + 1, f, lmax)
            # x == 2: no channel is dithered, so the colour is the same at lo.
            if lo < hi and x == 0 and self._over(lo) == 1:
                # Passing levels within an interval are contiguous (the drift
                # is convex along it): find where they start (to within the
                # integer drift's precision, without floats).
                while hi - lo > 1:
                    mid = (lo + hi) >> 1
                    if self._over(mid, False) == 1:
                        lo = mid
                    else:
                        hi = mid
                return self._level(hi, f, lmax)
            hi = lo - 1
            i -= 1
        return 0

    def _scale_for(self, c, v, a):
        """Smallest scale s at which channel c of value v outputs at least a
        (8.8 fixed point) before rounding."""
        st = self.stage
        if st.gamma is None:
            sc = ((a << 8) - 128 + v - 1) // v
        else:
            gt = st.gtab
            lo = 0
            hi = _GSIZE - 1
            if gt[hi] < a:
                return 1 << 30
            while lo < hi:
                mid = (lo + hi) >> 1
                if gt[mid] >= a:
                    hi = mid
                else:
                    lo = mid + 1
            sc = ((lo << 13) + v - 1) // v
        w = st.white
        if w is None or not w[c]:
            return sc
        return ((sc << _Q) - 2048 + w[c] - 1) // w[c]

    def _over(self, s, exact=True):
        """1 if the colour at scale s drifts beyond the threshold, else 0,
        or 2 when it is within it and no channel is dithered there. exact:
        decide close calls in floats (otherwise within about 5%)."""
        st = self.stage
        r, g, b = self._px
        w = st.white
        if w is None:
            s0 = s1 = s2 = s
        else:
            s0 = (s * w[0] + 2048) >> _Q
            s1 = (s * w[1] + 2048) >> _Q
            s2 = (s * w[2] + 2048) >> _Q
        gt = st.gtab if st.gamma is not None else None
        if gt is None:
            a0 = (r * s0 + 128) >> 8
            a1 = (g * s1 + 128) >> 8
            a2 = (b * s2 + 128) >> 8
        else:
            a0 = gt[(r * s0) >> 13]
            a1 = gt[(g * s1) >> 13]
            a2 = gt[(b * s2) >> 13]
        umax = st.umax
        # With gamma the target moves within an interval as well.
        whole = 2 if gt is None else 0
        if a0 >= umax:
            o0 = a0
            whole = 0
        else:
            o0 = ((a0 + 128) >> 8) << 8
        if a1 >= umax:
            o1 = a1
            whole = 0
        else:
            o1 = ((a1 + 128) >> 8) << 8
        if a2 >= umax:
            o2 = a2
            whole = 0
        else:
            o2 = ((a2 + 128) >> 8) << 8
        if (o0 == a0 and o1 == a1 and o2 == a2) or not (o0 or o1 or o2):
            return whole
        if gt is None:
            tu, tv = self._tuv
        else:
            tu, tv = _uvq(a0, a1, a2)
        # _uvq(o0, o1, o2), inline: no tuple to allocate.
        m = o0 if o0 > o1 else o1
        if o2 > m:
            m = o2
        x0 = o0
        x1 = o1
        x2 = o2
        while m > 8191:
            x0 = (x0 + 1) >> 1
            x1 = (x1 + 1) >> 1
            x2 = (x2 + 1) >> 1
            m = (m + 1) >> 1
        d = _KD0 * x0 + _KD1 * x1 + _KD2 * x2
        n = (_KU0 * x0 + _KU1 * x1 + _KU2 * x2) << 7
        q = n // d
        n = (n - q * d) << 6
        r = n // d
        du = (((q << 6) + r) << 2) + (((n - r * d) << 2) // d) - tu
        n = (_KV0 * x0 + _KV1 * x1 + _KV2 * x2) << 7
        q = n // d
        n = (n - q * d) << 6
        r = n // d
        dv = (((q << 6) + r) << 2) + (((n - r * d) << 2) // d) - tv
        e = du * du + dv * dv
        if not exact:
            return 1 if e > self._mid2 else whole
        if e < self._lo2:
            return whole
        if e > self._hi2:
            return 1
        # Close to the threshold: decide exactly, in floats (rare).
        if gt is None:
            t = self._t
        else:
            t = (a0, a1, a2)
        return 1 if drift((o0, o1, o2), t) > self._thr else whole

    @staticmethod
    def _level(s, f, lmax):
        """Lowest level (1/256 units) at which process() reaches scale s."""
        return min(lmax, max(0, (s * 4080 - 2032 + f - 1) // f))

try:
    from time import ticks_us as _ticks_us, ticks_diff as _ticks_diff
except ImportError:
    from time import perf_counter as _pc

    def _ticks_us():
        return int(_pc() * 1000000)

    def _ticks_diff(a, b):
        return a - b
