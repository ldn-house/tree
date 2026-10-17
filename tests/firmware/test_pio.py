"""The real PIO program in firmware/app/driver.py, assembled the way
rp2.asm_pio does it (only PIO names as globals) and simulated cycle by
cycle. Checks WS2811 pulse widths against the datasheet windows."""

import importlib.util
import sys
import types

import devsim

FW = devsim.ROOT / "firmware"


def load():
    spec = importlib.util.spec_from_file_location("rp2", devsim.MOCKS / "rp2.py")
    rp2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rp2)
    machine = types.ModuleType("machine")
    machine.Pin = lambda *a, **k: None
    saved = {k: sys.modules.get(k) for k in ("rp2", "machine")}
    sys.modules["rp2"] = rp2
    sys.modules["machine"] = machine
    sys.path.insert(0, str(FW))
    try:
        for k in [k for k in sys.modules if k == "app" or k.startswith("app.")]:
            del sys.modules[k]
        import app.driver as driver
    finally:
        sys.path.remove(str(FW))
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return rp2, driver


def bits_from(runs):
    """[(high_cycles, low_cycles), ...] per bit (the line idles low)."""
    out = []
    for i in range(0, len(runs) - 1, 2):
        assert runs[i][0] == 1 and runs[i + 1][0] == 0, runs[:6]
        out.append((runs[i][1], runs[i + 1][1]))
    return out


def test_ws2811_waveform():
    rp2, driver = load()
    prog = driver._ws2811
    assert isinstance(prog, rp2.Program)  # assembled without NameError
    ns = 1e9 / driver.PIO_FREQ
    data = bytes([0b10110010, 0x00, 0xFF, 0x5A])
    runs = rp2.simulate(prog, data)
    # The first segment is the initial low before the first bit; drop it.
    if runs[0][0] == 0:
        runs = runs[1:]
    bits = bits_from(runs + [(0, 10**6)])
    want = [int(c) for b in data for c in format(b, "08b")]
    assert len(bits) == len(want)
    for (hi, lo), bit in zip(bits[:-1], want[:-1]):
        assert (hi + lo) * ns == driver.BIT_NS, (hi, lo)  # 800 kHz exactly
        if bit:
            assert 650 <= hi * ns <= 950 and 450 <= hi * ns, (bit, hi * ns)
            assert 300 <= lo * ns <= 650
        else:
            # Inside both the original WS2811 high-speed window (100-400 ns)
            # and the WS2812-style one (250-550 ns), with >= 50 ns margin to
            # the WS2812-style minimum (the old program's 250 ns had none).
            assert 300 <= hi * ns <= 400, (bit, hi * ns)
            assert 700 <= lo * ns <= 1000
    widths = {b: hi * ns for (hi, _), b in zip(bits, want)}
    print("T0H %.0f ns, T1H %.0f ns" % (widths[0], widths[1]))
    assert widths[0] == 350 and widths[1] == 800
