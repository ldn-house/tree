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


def averages(o, src, brightness, frames):
    """Mean output per channel position (r, g, b) over frames and LEDs."""
    sums = [0, 0, 0]
    totals = []
    dst = bytearray(len(src))
    for _ in range(frames):
        o.process(src, dst, brightness)
        totals.append(sum(dst))
        for c in range(3):
            sums[c] += sum(dst[c::3])
    n = len(src) // 3
    return [s / frames / n for s in sums], totals


@pytest.mark.parametrize("percent", [2, 3, 4, 5])
def test_warm_white_keeps_its_hue_when_dim(percent):
    """At 2-5% the whole-tree cap leaves warm white's green and blue below
    1 LSB. Flooring made them 0 (pure red); dithering must keep their
    average, in the right ratio to red."""
    b = round(percent * 255 / 100)
    o = _output.Output(N)
    avg, _ = averages(o, solid(WARM), b, 64)
    want = [exact(o, c, b) for c in WARM]
    assert avg[1] > 0 and avg[2] > 0
    for got, w in zip(avg, want):
        assert got == pytest.approx(w, rel=0.03, abs=0.01)
    assert avg[1] / avg[0] == pytest.approx(140 / 255, rel=0.03)
    assert avg[2] / avg[0] == pytest.approx(50 / 255, rel=0.05)


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


def test_dither_average_matches_target():
    o = _output.Output(N, budget=10**9)
    src = bytearray((k * 37) & 255 for k in range(3 * N))
    for b in (2, 9, 77):
        dst = bytearray(3 * N)
        sums = [0] * (3 * N)
        frames = 256
        for _ in range(frames):
            o.process(src, dst, b)
            for k in range(3 * N):
                sums[k] += dst[k]
        for k in range(3 * N):
            assert sums[k] / frames == pytest.approx(src[k] * b / 255, abs=0.01)


def test_whole_levels_do_not_change():
    """A channel whose target is a whole number is sent unchanged every
    frame (no dither noise on exact levels)."""
    o = _output.Output(N, budget=10**9)
    src = bytearray((k * 13) & 255 for k in range(3 * N))
    dst = bytearray(3 * N)
    for _ in range(10):
        o.process(src, dst, 255)
        assert dst == src


@pytest.mark.parametrize("brightness", [255, 60, 8])
def test_power_cap_holds(brightness):
    """Dense frames, capped: no frame is over the budget (rounding and
    dithering are trimmed at the cap), and the average stays at the target."""
    o = _output.Output(N)
    src = bytearray((k * 101 + 7) & 255 for k in range(3 * N))
    dst = bytearray(3 * N)
    totals = []
    frames = 64
    for _ in range(frames):
        o.process(src, dst, brightness)
        total = sum(dst)
        assert total <= o.budget
        totals.append(total)
    target = sum(exact(o, v, brightness) for v in src)
    assert target <= o.budget
    assert sum(totals) / frames <= target + 3 * N / frames + 1
    assert sum(totals) / frames >= target * 0.995


def test_tree_does_not_flicker_in_step():
    """Per-pixel phases: at 3% warm white the total output barely moves from
    frame to frame, even though single LEDs toggle."""
    o = _output.Output(N)
    _, totals = averages(o, solid(WARM), 8, 64)
    mean = sum(totals) / len(totals)
    spread = max(totals) - min(totals)
    assert spread <= 0.02 * mean + 3


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
