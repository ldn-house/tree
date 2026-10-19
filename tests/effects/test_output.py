"""Output stage (effects/_output.py) behaviour at low brightness.

    uv run --with pytest pytest tests/effects

The native (viper) kernels only exist on MicroPython; tools/output_check.py
checks they match these plain-Python ones byte for byte.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from effects import _output  # noqa: E402

N = 500
WARM = (255, 140, 50)


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
    o = _output.Output(N)
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
        o = _output.Output(N)
        frames = frames_of(o, src, b, 40)
        assert all(f == frames[0] for f in frames)


@pytest.mark.parametrize("percent", [1, 3, 20, 60])
def test_each_led_within_one_of_exact(percent):
    b = round(percent * 255 / 100)
    src = bytearray((k * 37 + 11) & 255 for k in range(3 * N))
    o = _output.Output(N)
    (frame,) = frames_of(o, src, b, 1)
    for k in range(3 * N):
        assert abs(frame[k] - exact(o, src[k], b)) < 1, (k, frame[k], exact(o, src[k], b))


def test_spatial_average_matches_target_for_any_fraction():
    """Across the tree, a uniform target with any fractional part averages
    to that target (the thresholds are unbiased)."""
    o = _output.Output(N, budget=10**9)
    for b in range(1, 256, 5):
        (frame,) = frames_of(o, solid((255, 128, 40)), b, 1)
        for c, v in enumerate((255, 128, 40)):
            assert sum(frame[c::3]) / N == pytest.approx(v * b / 255, abs=0.02)


def test_old_floor_collapsed_to_red():
    """The behaviour this replaces, for the record: at 3% the old stage
    sent (1, 0, 0) for warm white."""
    o = _output.Output(N, dither=False)
    dst = bytearray(3 * N)
    o.process(solid(WARM), dst, 8)
    floored = tuple(int(exact(o, c, 8)) for c in WARM)
    assert floored == (1, 0, 0)
    assert tuple(dst[:3]) == tuple(int(exact(o, c, 8) + 0.5) for c in WARM) == (2, 1, 0)


def test_rounds_to_nearest_without_dither():
    o = _output.Output(N, budget=10**9, dither=False)
    src = bytearray(range(256)) + bytearray(3 * N - 256)
    dst = bytearray(3 * N)
    for b in (1, 3, 7, 40, 128, 200, 255):
        o.process(src, dst, b)
        for v in range(256):
            assert abs(dst[v] - v * b / 255) <= 0.5 + 1 / 128, (b, v, dst[v])


def test_dither_modes():
    assert _output.Output(N).dither == "spatial"
    assert _output.Output(N, dither=True).dither == "spatial"
    assert _output.Output(N, dither=False).dither is None
    assert _output.Output(N, dither="temporal").dither == "temporal"
    with pytest.raises(ValueError):
        _output.Output(N, dither="bayer")


def test_temporal_opt_in_averages_over_time():
    """The opt-in temporal mode still averages exactly per LED over time."""
    o = _output.Output(N, budget=10**9, dither="temporal")
    src = bytearray((k * 37) & 255 for k in range(3 * N))
    for b in (2, 9, 77):
        frames = frames_of(o, src, b, 256)
        for k in range(0, 3 * N, 7):
            assert sum(f[k] for f in frames) / 256 == pytest.approx(src[k] * b / 255, abs=0.01)


@pytest.mark.parametrize("dither", ["spatial", "temporal", False])
def test_whole_levels_do_not_change(dither):
    """A channel whose target is a whole number is sent unchanged."""
    o = _output.Output(N, budget=10**9, dither=dither)
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
    o = _output.Output(N, dither=dither)
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
    lin = _output.Output(N, budget=10**9, dither=False)
    gam = _output.Output(N, budget=10**9, dither=False, gamma=2.2)
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
    o = _output.Output(N, budget=10**9, dither=False, white=(1.0, 0.5, 0.25))
    dst = bytearray(3 * N)
    o.process(solid((200, 200, 200)), dst, 255)
    assert tuple(dst[:3]) == (200, 100, 50)
    g = _output.Output(N, budget=10**9, dither=False, white=(1.0, 0.5, 0.25), order="GRB")
    g.process(solid((200, 200, 200)), dst, 255)
    assert tuple(dst[:3]) == (100, 200, 50)


def test_colour_order():
    o = _output.Output(N, budget=10**9, order="GRB")
    dst = bytearray(3 * N)
    o.process(solid((10, 20, 30)), dst, 255)
    assert tuple(dst[:3]) == (20, 10, 30)


def test_brightness_zero_is_dark():
    o = _output.Output(N)
    dst = bytearray(3 * N)
    for _ in range(5):
        o.process(solid(WARM), dst, 0)
        assert not any(dst)
