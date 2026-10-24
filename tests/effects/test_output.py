"""Output stage (effects/_output.py) behaviour at low brightness.

    uv run --with pytest pytest tests/effects

The native (viper) kernels only exist on MicroPython; tools/output_check.py
checks they match these plain-Python ones byte for byte.
"""

import sys
from pathlib import Path

import pytest
from array import array

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from effects import _output  # noqa: E402

N = 500
WARM = (255, 140, 50)


def stage(**kw):
    """An output stage without the brightness floor and dithering at every
    level, for testing rounding and dithering themselves at raw brightness
    levels (the floor and the undithered low end have their own tests)."""
    kw.setdefault("floor", 0)
    kw.setdefault("dither_min", 0)
    return _output.Output(N, **kw)


def solid(color, n=N):
    return bytearray(color * n)


def exact(o, value, brightness):
    """The exact (unquantised) linear output for a raw value."""
    return value * o.factor * brightness / 255 / 4096


def frames_of(o, src, brightness, count):
    out = []
    dst = bytearray(len(src))
    for _ in range(count):
        o.process(src, dst, brightness)
        out.append(bytes(dst))
    return out


@pytest.mark.parametrize("percent", [1, 2, 3, 4, 5, 10, 20, 25])
def test_warm_white_keeps_its_hue_when_dim(percent):
    """At low brightness the whole-tree cap leaves warm white's green and
    blue below 1 LSB. Flooring made them 0 (pure red). With spatial
    dithering a single frame's tree-wide average keeps every channel, in the
    right ratio to red."""
    b = round(percent * 255 / 100)
    o = stage()
    (frame,) = frames_of(o, solid(WARM), b, 1)
    avg = [sum(frame[c::3]) / N for c in range(3)]
    want = [exact(o, c, b) for c in WARM]
    assert avg[1] > 0 and avg[2] > 0
    for got, w in zip(avg, want):
        assert got == pytest.approx(w, rel=0.03, abs=0.01)
    assert avg[1] / avg[0] == pytest.approx(140 / 255, rel=0.03)
    assert avg[2] / avg[0] == pytest.approx(50 / 255, rel=0.05)


@pytest.mark.parametrize("percent", [1, 3, 20, 100])
def test_static_frame_is_identical_every_frame(percent):
    """Spatial dithering is fixed per LED: a static colour sends exactly the
    same bytes on every frame, so nothing flickers."""
    b = round(percent * 255 / 100)
    for src in (solid(WARM), bytearray((k * 37 + 11) & 255 for k in range(3 * N))):
        o = stage()
        frames = frames_of(o, src, b, 40)
        assert all(f == frames[0] for f in frames)


@pytest.mark.parametrize("percent", [1, 3, 20, 60])
def test_each_led_within_one_of_exact(percent):
    b = round(percent * 255 / 100)
    src = bytearray((k * 37 + 11) & 255 for k in range(3 * N))
    o = stage()
    (frame,) = frames_of(o, src, b, 1)
    for k in range(3 * N):
        assert abs(frame[k] - exact(o, src[k], b)) < 1, (k, frame[k], exact(o, src[k], b))


def test_spatial_average_matches_target_for_any_fraction():
    """Across the tree, a uniform target with any fractional part averages
    to that target (the thresholds are unbiased)."""
    o = stage(budget=10**9)
    for b in range(1, 256, 5):
        (frame,) = frames_of(o, solid((255, 128, 40)), b, 1)
        for c, v in enumerate((255, 128, 40)):
            assert sum(frame[c::3]) / N == pytest.approx(v * b / 255, abs=0.02)


def test_old_floor_collapsed_to_red():
    """The behaviour this replaces, for the record: at 3% the old stage
    sent (1, 0, 0) for warm white."""
    o = stage(dither=False)
    dst = bytearray(3 * N)
    o.process(solid(WARM), dst, 8)
    floored = tuple(int(exact(o, c, 8)) for c in WARM)
    assert floored == (1, 0, 0)
    assert tuple(dst[:3]) == tuple(int(exact(o, c, 8) + 0.5) for c in WARM) == (2, 1, 0)


def test_rounds_to_nearest_without_dither():
    o = stage(budget=10**9, dither=False)
    src = bytearray(range(256)) + bytearray(3 * N - 256)
    dst = bytearray(3 * N)
    for b in (1, 3, 7, 40, 128, 200, 255):
        o.process(src, dst, b)
        for v in range(256):
            assert abs(dst[v] - v * b / 255) <= 0.5 + 1 / 128, (b, v, dst[v])


def test_dither_modes():
    assert _output.Output(N).dither == "spatial"
    assert stage(dither=True).dither == "spatial"
    assert stage(dither=False).dither is None
    assert stage(dither="temporal").dither == "temporal"
    with pytest.raises(ValueError):
        stage(dither="bayer")


def test_temporal_opt_in_averages_over_time():
    """The opt-in temporal mode still averages exactly per LED over time."""
    o = stage(budget=10**9, dither="temporal")
    src = bytearray((k * 37) & 255 for k in range(3 * N))
    for b in (2, 9, 77):
        frames = frames_of(o, src, b, 256)
        for k in range(0, 3 * N, 7):
            assert sum(f[k] for f in frames) / 256 == pytest.approx(src[k] * b / 255, abs=0.01)


@pytest.mark.parametrize("dither", ["spatial", "temporal", False])
def test_whole_levels_do_not_change(dither):
    """A channel whose target is a whole number is sent unchanged."""
    o = stage(budget=10**9, dither=dither)
    src = bytearray((k * 13) & 255 for k in range(3 * N))
    dst = bytearray(3 * N)
    for _ in range(10):
        o.process(src, dst, 255)
        assert dst == src


@pytest.mark.parametrize("dither", ["spatial", "temporal", False])
@pytest.mark.parametrize("brightness", [255, 60, 8])
def test_power_cap_holds(brightness, dither):
    """Dense frames, capped: no frame is over the budget (rounding and
    dithering are trimmed at the cap), and the average stays at the target."""
    o = stage(dither=dither)
    src = bytearray((k * 101 + 7) & 255 for k in range(3 * N))
    frames = frames_of(o, src, brightness, 16)
    totals = [sum(f) for f in frames]
    assert max(totals) <= o.budget
    target = sum(exact(o, v, brightness) for v in src)
    assert target <= o.budget
    # Rounding errors on arbitrary data partly cancel; with one threshold
    # shared by an LED's channels the total for a single static frame can be
    # off by a few tens of units (about 1 per 40 lit channels).
    assert sum(totals) / len(totals) >= target - max(0.005 * target, 40)
    if dither != "temporal":
        assert all(f == frames[0] for f in frames)


def test_gamma_never_above_linear():
    lin = stage(budget=10**9, dither=False)
    gam = stage(budget=10**9, dither=False, gamma=2.2)
    src = bytearray(k & 255 for k in range(3 * N))
    a = bytearray(3 * N)
    b = bytearray(3 * N)
    for br in (5, 100, 255):
        lin.process(src, a, br)
        gam.process(src, b, br)
        assert all(y <= x for x, y in zip(a, b))
    gam.process(bytearray([255] * (3 * N)), b, 255)
    assert b[0] == 255


def test_white_balance_scales_source_channels():
    o = stage(budget=10**9, dither=False, white=(1.0, 0.5, 0.25))
    dst = bytearray(3 * N)
    o.process(solid((200, 200, 200)), dst, 255)
    assert tuple(dst[:3]) == (200, 100, 50)
    g = stage(budget=10**9, dither=False, white=(1.0, 0.5, 0.25), order="GRB")
    g.process(solid((200, 200, 200)), dst, 255)
    assert tuple(dst[:3]) == (100, 200, 50)


def test_colour_order():
    o = stage(budget=10**9, order="GRB")
    dst = bytearray(3 * N)
    o.process(solid((10, 20, 30)), dst, 255)
    assert tuple(dst[:3]) == (20, 10, 30)


def test_brightness_zero_is_dark():
    o = stage()
    dst = bytearray(3 * N)
    for _ in range(5):
        o.process(solid(WARM), dst, 0)
        assert not any(dst)


# -- the low end: brightness floor, no dithering below DITHER_MIN --
#
# These pin the wire bytes. They can't show how an LED looks: at a few levels
# a WS2811's three dies aren't in their linear proportions (at 1/255 they look
# about equally bright, so blue 1 next to blue 0 reads as white next to
# yellow), and the eye blends neighbouring LEDs differently depending on how
# densely they sit in the tree. That needs the real tree.

TONES = ((255, 140, 50), (255, 119, 35), (255, 87, 14), (255, 68, 7))
TONE2 = TONES[2]  # warm_white's default


def hass(pct):
    """Home Assistant's brightness for a slider percentage (brightness_scale 255)."""
    return round(pct * 255 / 100)


def targets(o, color, brightness):
    """Exact per-channel output for a solid frame, through the floor."""
    return [c * o.factor / 4096 * o.level(brightness) / 65280 for c in color]


def leds(frame):
    return [tuple(frame[3 * i:3 * i + 3]) for i in range(len(frame) // 3)]


def test_defaults():
    o = _output.Output(N)
    assert o.floor == _output.FLOOR_PCT == 4
    assert o.dither_min == _output.DITHER_MIN == 9


def test_floor_mapping():
    """0 stays off, HA's 1% lands on the floor, 100% is unchanged, and
    every HA step between is a higher level."""
    o = _output.Output(N)
    levels = [o.level(b) for b in range(256)]
    assert levels[0] == 0
    assert levels[255] == 65280
    assert all(b >= a for a, b in zip(levels, levels[1:]))
    floor_q = round(4 * 65280 / 100)
    assert levels[1] == levels[2] == levels[hass(1)] == floor_q
    assert all(b > a for a, b in zip(levels[3:], levels[4:]))
    assert o.level(129) / 652.8 == pytest.approx(4 + 96 * 126 / 252, abs=0.01)


def test_no_floor_is_identity():
    o = stage()
    assert [o.level(b) for b in range(256)] == [b * 256 for b in range(256)]


def test_set_floor():
    o = _output.Output(N)
    o.set_floor(0)
    assert o.level(1) == 256
    o.set_floor(50)
    assert o.level(3) == 65280 // 2
    o.set_floor(-5)
    assert o.floor == 0
    o.set_floor(500)
    assert o.floor == 100 and o.level(1) == 65280


@pytest.mark.parametrize("tone", range(4))
@pytest.mark.parametrize("dither", ["spatial", "temporal"])
def test_no_spatial_split_below_dither_min(tone, dither):
    """For every warm white tone at every HA step, a channel whose target is
    below DITHER_MIN is the same on every LED (and stays so frame after
    frame), rounded to nearest; a channel at or above it is dithered within
    1 of exact and averages to the target."""
    o = _output.Output(N, dither=dither)
    dst = bytearray(3 * N)
    for pct in range(1, 101):
        b = hass(pct)
        for _ in range(3):
            o.process(solid(TONES[tone]), dst, b)
            if sum(dst) == o.budget:
                continue  # rounding went over the cap and was trimmed (near 100% only)
            want = targets(o, TONES[tone], b)
            for c, x in enumerate(want):
                vals = dst[c::3]
                if x < _output.DITHER_MIN - 0.01:
                    assert len(set(vals)) == 1, (pct, c, x, set(vals))
                    assert abs(vals[0] - x) <= 0.5 + 1 / 64, (pct, c, x, vals[0])
                elif x > _output.DITHER_MIN + 0.01:
                    assert all(abs(v - x) < 1 for v in vals), (pct, c, x)
                    if dither == "spatial":
                        assert sum(vals) / N == pytest.approx(x, abs=0.02)


def test_warm_white_bottom_of_the_slider():
    """At HA's 1%, each tone is one colour on every LED. Tone 2 (the
    default) is (3, 1, 0): green/red 0.33 against its 0.34, the closest
    whole-number colour; one level lower would be (2, 1, 0), too yellow."""
    dst = bytearray(3 * N)
    got = []
    for c in TONES:
        _output.Output(N).process(solid(c), dst, hass(1))
        assert len(set(leds(dst))) == 1
        got.append(tuple(dst[:3]))
    assert got == [(2, 1, 0), (2, 1, 0), (3, 1, 0), (3, 1, 0)]
    _output.Output(N, floor=3).process(solid(TONE2), dst, hass(1))
    assert tuple(dst[:3]) == (2, 1, 0)


def test_one_colour_until_red_is_dithered():
    """Warm white (tone 2) is a single colour on the whole tree from 1% until
    red's target reaches DITHER_MIN; above that only red differs between
    LEDs (by one level of at least 9), while green and blue stay uniform
    until they reach DITHER_MIN too."""
    o = _output.Output(N)
    dst = bytearray(3 * N)
    seen_red_split = False
    for pct in range(1, 101):
        b = hass(pct)
        o.process(solid(TONE2), dst, b)
        if sum(dst) == o.budget:
            continue  # trimmed at the cap (near 100% only)
        want = targets(o, TONE2, b)
        colours = set(leds(dst))
        if want[0] < _output.DITHER_MIN - 0.01:
            assert len(colours) == 1, (pct, colours)
        else:
            for c in (1, 2):
                if want[c] < _output.DITHER_MIN - 0.01:
                    assert len({x[c] for x in colours}) == 1, (pct, colours)
            reds = {c[0] for c in colours}
            assert min(reds) >= _output.DITHER_MIN and max(reds) - min(reds) <= 1
            seen_red_split = seen_red_split or len(reds) == 2
    assert seen_red_split


def test_dims_monotonically():
    """Whole-tree warm white, capped: the output never gets brighter as the
    slider goes down. Below DITHER_MIN neighbouring steps can give the same
    colour (the levels are whole numbers); once red is dithered, every step
    is brighter."""
    o = _output.Output(N)
    dst = bytearray(3 * N)
    totals = []
    for b in range(3, 256):
        o.process(solid(TONE2), dst, b)
        assert o.capped
        totals.append((b, sum(dst), targets(o, TONE2, b)[0]))
    assert all(t2 >= t1 for (_, t1, _), (_, t2, _) in zip(totals, totals[1:]))
    dithered = [(b, t) for b, t, red in totals if red >= _output.DITHER_MIN + 0.1]
    assert all(t2 > t1 for (_, t1), (_, t2) in zip(dithered, dithered[1:]))


@pytest.mark.parametrize("dither", ["spatial", "temporal", False])
@pytest.mark.parametrize("brightness", [1, 3, 8, 40])
def test_power_cap_holds_at_the_low_end(brightness, dither):
    o = _output.Output(N, dither=dither)
    for src in (solid((255, 255, 255)), bytearray((k * 101 + 7) & 255 for k in range(3 * N))):
        for f in frames_of(o, src, brightness, 8):
            assert sum(f) <= o.budget


def test_textured_frames_are_static_and_close():
    """Arbitrary content at low brightness: identical bytes every frame and
    every channel within 1 of exact (nearest below DITHER_MIN)."""
    src = bytearray((k * 37 + 11) & 255 for k in range(3 * N))
    o = _output.Output(N, budget=10**9)
    for b in (3, 20, 60):
        frames = frames_of(o, src, b, 5)
        assert all(f == frames[0] for f in frames)
        scale = o.level(b) / 65280
        for k in range(3 * N):
            x = src[k] * scale
            assert abs(frames[0][k] - x) <= (0.5 + 1 / 64 if x < _output.DITHER_MIN else 1), (b, k, x)


def test_full_brightness_unchanged_by_floor():
    """Brightness 255 (what calibration frames use) is the same with or
    without the floor and the undithered low end, so calibration frames are
    byte-identical."""
    src = bytearray((k * 13) & 255 for k in range(3 * N))
    a = bytearray(3 * N)
    b = bytearray(3 * N)
    stage(dither=False).process(src, a, 255)
    _output.Output(N, dither=False).process(src, b, 255)
    assert a == b
    _output.Output(N, budget=10**9).process(src, b, 255)
    assert b == src


# ─── Colour floor ─────────────────────────────────────────────────────────────

TONE2 = (255, 87, 14)  # warm_white's default tone


def colour_floor(color, accuracy=_output.COLOUR_ACCURACY, **kw):
    """(stage, ColourFloor) after analysing a solid frame of color."""
    o = _output.Output(N, **kw)
    cf = _output.ColourFloor(o, accuracy)
    cf.add(solid(color))
    return o, cf


def bottom(o, color):
    """The wire colours at Home Assistant's 1% (brightness 3)."""
    (frame,) = frames_of(o, solid(color), 3, 1)
    return {tuple(frame[k:k + 3]) for k in range(0, len(frame), 3)}


def worst_drift(o, color, top_pct=_output.COLOUR_FLOOR_MAX):
    """Largest drift at any brightness from the bottom up to top_pct, with
    dithered channels at their exact value (the model ColourFloor uses)."""
    luts = [array("H", bytes(512)), array("H", bytes(512))]
    worst = 0
    for b in range(1, 256):
        if o.level(b) > top_pct * 652.8:
            break
        o.process(solid(color), bytearray(3 * N), b)
        s = (o.factor * o.level(b) + 2032) // 4080
        _output._lut_py(luts[0], s, o.gtab, 0, o.umax)
        _output._lut_py(luts[1], s, o.gtab, 0, 0)
        out = [luts[0][c] for c in color]
        exact = [c * s / 256 for c in color]
        worst = max(worst, _output.drift(out, exact))
    return worst


def test_drift_matches_the_tree():
    """The model against what was seen on the tree (23 Oct): warm white tone
    2 at (3, 1, 0) and (5, 2, 0) looked yellow, (22, 8, 1) warm white."""
    assert _output.drift((3, 1, 0), TONE2) == pytest.approx(0.012, abs=0.0005)
    assert _output.drift((5, 2, 0), TONE2) == pytest.approx(0.016, abs=0.0005)
    assert _output.drift((22, 8, 1), TONE2) == pytest.approx(0.005, abs=0.0005)
    assert _output.drift((0, 0, 0), TONE2) == 0  # dark is no colour
    assert _output.drift((7, 0, 0), (255, 0, 0)) == pytest.approx(0)


def test_warm_white_is_not_yellow_at_the_bottom():
    """At the default accuracy, warm white's bottom keeps its blue: no more
    of the blue-less yellow that #25's 4% floor gave, (3, 1, 0)."""
    o, cf = colour_floor(TONE2)
    assert cf.pct > _output.FLOOR_PCT and o.floor_pct() == cf.pct
    colours = bottom(o, TONE2)
    assert all(b == 1 and g == 4 and 10 <= r <= 11 for r, g, b in colours), colours
    assert worst_drift(o, TONE2) <= _output.COLOUR_ACCURACY
    # Nowhere dimmer would do: one step below the floor is still yellow.
    o.set_colour_floor(cf.level - 64)
    assert worst_drift(o, TONE2) > _output.COLOUR_ACCURACY


@pytest.mark.parametrize("color", [TONE2, (255, 140, 50), (255, 119, 35), (255, 68, 7), (255, 60, 120),
                                   (120, 0, 255), (40, 200, 90)])
def test_stricter_accuracy_raises_the_floor(color):
    floors = []
    for acc in (0.05, 0.02, 0.01, 0.005):
        o, cf = colour_floor(color, acc)
        # (worst_drift's targets are in 1/256 levels: within 1%.)
        assert worst_drift(o, color) <= _output.tolerance(color, acc) * 1.01
        floors.append(cf.pct)
    assert floors == sorted(floors) and floors[0] < floors[-1], floors


def test_floor_values_for_warm_white():
    """Pinned: the floor each warm_white tone gets under the 1.5 A cap (red
    at 100% is 55-74 levels), at 0.02 and 0.01. Tone 3, more orange than
    COLOUR_HUE_FROM, gets a little more tolerance (lower at 0.02)."""
    got = {}
    for tone in ((255, 140, 50), (255, 119, 35), TONE2, (255, 68, 7)):
        got[tone] = tuple(colour_floor(tone, acc)[1].pct for acc in (0.02, 0.01))
    assert got == {(255, 140, 50): (8.31, 15.52), (255, 119, 35): (6.09, 16.18),
                   TONE2: (10.95, 15.79), (255, 68, 7): (7.61, 22.85)}, got


@pytest.mark.parametrize("color", [(255, 0, 0), (0, 255, 0), (0, 0, 40), (255, 255, 255), (9, 9, 9)])
def test_single_channels_and_greys_never_hold_the_floor_up(color):
    o, cf = colour_floor(color, 0.005)
    assert cf.pct == 0 and o.floor_pct() == _output.FLOOR_PCT


def test_christmas_frame_has_no_colour_floor():
    o = _output.Output(N)
    cf = _output.ColourFloor(o)
    cf.add(bytearray((255, 0, 0, 0, 255, 0) * (N // 2)))
    assert cf.pct == 0


def test_mixed_frames():
    """The floor is what 90% of the lit pixels need; pixels below half the
    frame's brightest don't count."""
    # Half warm white: it sets the floor, and keeps its blue at 1%.
    o = _output.Output(N)
    cf = _output.ColourFloor(o)
    half = bytearray((TONE2 + (255, 0, 0)) * (N // 2))
    cf.add(half)
    assert cf.pct > 10
    (frame,) = frames_of(o, half, 3, 1)
    assert {frame[k + 2] for k in range(0, 3 * N, 6)} == {1}
    # A few warm pixels among red ones (1 in 32 analysed): under 10%, so
    # they don't.
    cf.reset()
    frame = bytearray((255, 0, 0) * N)
    for i in range(3, N, 100):
        frame[3 * i:3 * i + 3] = bytes(TONE2)
    cf.add(frame)
    assert cf.pct == 0
    # Dim warm pixels (tails, fades) don't count either.
    cf.reset()
    cf.add(bytearray(((120, 41, 7) + (255, 0, 0)) * (N // 2)))
    assert cf.pct == 0
    # Black frame: nothing to analyse.
    cf.reset()
    cf.add(bytearray(3 * N))
    assert cf.pct == 0 and cf.pixels == 0


def test_animated_frames_are_pooled_and_reset_starts_over():
    o = _output.Output(N)
    cf = _output.ColourFloor(o)
    assert cf.due(0) and not cf.due(-1)
    cf.add(bytearray((255, 0, 0) * N))
    assert cf.pct == 0 and not cf.due(999) and cf.due(1000)
    cf.add(bytearray(TONE2 * N))  # half the pooled pixels are now warm
    assert cf.pct > 10 and cf.pixels == 2 * _output.COLOUR_SAMPLES
    cf.add(bytearray(TONE2 * N))
    cf.add(bytearray(TONE2 * N))
    assert not cf.due(10 ** 6)  # four frames, then nothing until a reset
    cf.reset()
    assert cf.due(0)
    cf.add(bytearray((0, 255, 0) * N))
    assert cf.pct == 0 and o.floor_pct() == _output.FLOOR_PCT


def test_step_by_step_matches_all_at_once():
    frame = bytearray()
    for i in range(N):
        frame += bytes(((i * 7) & 255, (i * 13 + 90) & 255, (255 - i) & 255))
    a = _output.ColourFloor(_output.Output(N))
    a.add(frame)
    b = _output.ColourFloor(_output.Output(N))
    b.start(frame)
    steps = 1
    while not b.step(3):
        steps += 1
    assert steps > 5 and (b.level, b.pixels) == (a.level, a.pixels) and a.level > 0


def test_accuracy_bounds_and_off():
    assert _output.colour_accuracy(0) == 0 and _output.colour_accuracy(-1) == 0
    assert _output.colour_accuracy(0.001) == _output.COLOUR_MIN
    assert _output.colour_accuracy(1) == _output.COLOUR_MAX
    o, cf = colour_floor(TONE2, 0)
    assert cf.pct == 0 and o.floor_pct() == _output.FLOOR_PCT
    assert bottom(o, TONE2) == {(3, 1, 0)}  # #25's floor alone
    cf.set_accuracy(0.01)
    cf.add(solid(TONE2))
    assert cf.pct > 10


def test_user_floor_above_the_colour_floor_wins():
    o, cf = colour_floor(TONE2)
    o.set_floor(25)
    assert o.floor_pct() == 25 and o.level(3) == int(25 * 652.8 + 0.5)
    o.set_floor(0)
    assert o.floor_pct() == cf.pct and o.level(1) == cf.level
    assert o.level(0) == 0 and o.level(255) == 65280


def test_never_above_the_maximum():
    o, cf = colour_floor((255, 3, 1), 0.005)
    assert 0 < cf.pct <= _output.COLOUR_FLOOR_MAX


@pytest.mark.parametrize("kw", [{}, {"white": (1.0, 0.85, 0.7)}, {"gamma": 2.2}, {"dither_min": 0},
                                {"budget": 200000}])
def test_matches_brute_force(kw):
    """The boundary search against every scale process() can use: no level
    above the floor drifts, and the level just below it does."""
    for color in (TONE2, (255, 60, 120), (120, 0, 255), (10, 20, 255)):
        o = _output.Output(N, **kw)
        total = sum(color) * N
        f = (o.budget << 12) // total if total > o.budget else 4096
        lmax = _output.COLOUR_FLOOR_MAX * 65280 // 100
        smax = (f * lmax + 2032) // 4080
        drifts = [0.0]
        for s in range(1, smax + 1):
            sc = [s] * 3 if o.white is None else [(s * w + 2048) >> 12 for w in o.white]
            # As _lut_py() computes one entry, before and after rounding.
            exact = [o.gtab[(v * x) >> 13] if o.gamma else (v * x + 128) >> 8 for v, x in zip(color, sc)]
            out = [a if a >= o.umax else ((a + 128) >> 8) << 8 for a in exact]
            drifts.append(_output.drift(out, exact))
        t = color if o.white is None else [v * w for v, w in zip(color, o.white)]
        if o.gamma:
            t = [v ** o.gamma for v in t]
        for acc0 in (0.02, 0.01, 0.005):
            o, cf = colour_floor(color, acc0, **kw)
            acc = _output.tolerance(t, acc0)
            # The brute force's targets are the stage's 1/256-level ones and
            # the search uses exact targets, so compare within 2% of acc.
            need = (f * cf.level + 2032) // 4080 if cf.level else 1
            if cf.level < lmax:
                assert max(drifts[need:]) <= acc * 1.02, (kw, color, acc)
            if cf.level:
                # (A level maps to a scale within a couple of steps.)
                assert max(drifts[max(1, need - 3):need]) > acc * 0.98 or cf.level == lmax, (kw, color, acc)


def test_saturated_colours_get_a_hue_angle(monkeypatch):
    """Beyond COLOUR_HUE_FROM from white the allowed drift grows with the
    distance: the warm whites keep the fixed threshold (tones 0-2), and a
    nearly pure rainbow hue, which a fixed distance held at about 30-40%,
    gets a much lower floor."""
    acc = _output.COLOUR_ACCURACY
    assert _output.tolerance(TONE2, acc) == acc and _output.tolerance((255, 140, 50), acc) == acc
    assert _output.tolerance((255, 255, 255), acc) == acc and _output.tolerance((0, 0, 0), acc) == acc
    assert _output.tolerance((240, 15, 0), acc) > 1.8 * acc
    hue = {c: colour_floor(c)[1].pct for c in ((240, 15, 0), (0, 18, 237), TONE2, (255, 140, 50))}
    monkeypatch.setattr(_output, "COLOUR_HUE_FROM", 10)  # a fixed distance everywhere
    fixed = {c: colour_floor(c)[1].pct for c in hue}
    assert hue[TONE2] == fixed[TONE2] and hue[(255, 140, 50)] == fixed[(255, 140, 50)]
    assert fixed[(240, 15, 0)] > 25 and hue[(240, 15, 0)] < 15, (hue, fixed)
    assert fixed[(0, 18, 237)] > 30 and hue[(0, 18, 237)] < 15, (hue, fixed)


def test_step_with_a_time_limit_matches_all_at_once():
    """step(us=) stops once the time is up: with 1 us each call scans one
    chunk of pixels or analyses one new colour, and the result is the same
    as add()'s."""
    frame = bytearray()
    for i in range(N):
        frame += bytes(((i * 7) & 255, (i * 13 + 90) & 255, (255 - i) & 255))
    a = _output.ColourFloor(_output.Output(N))
    a.add(frame)
    b = _output.ColourFloor(_output.Output(N))
    b.start(frame)
    calls = 1
    while not b.step(us=1):
        calls += 1
    assert (b.level, b.pixels) == (a.level, a.pixels) and a.level > 0
    # N / 50 scan chunks, then one call per distinct colour.
    assert calls >= N // 50 + a.pixels - 1, calls
    # The previous floor stays until a started frame is done.
    b.reset()
    b.start(bytearray((255, 0, 0) * N))
    assert b.step(us=1) is False and b.stage.colour_floor == a.pct
    while not b.step(us=1):
        pass
    assert b.stage.colour_floor == 0
