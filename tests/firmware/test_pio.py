"""The real PIO program in firmware/app/driver.py, assembled the way
rp2.asm_pio does it (only PIO names as globals) and simulated cycle by
cycle, for the default and configured LED_TIMING_NS."""

import importlib.util
import sys
import types

import pytest

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


def waveform(rp2, driver, timing, data):
    """[(high_ns, low_ns, bit)] for each bit of data."""
    prog = driver.program(timing)
    assert isinstance(prog, rp2.Program)  # assembled without NameError
    ns = 1e9 / driver.PIO_FREQ
    runs = rp2.simulate(prog, data)
    if runs[0][0] == 0:
        runs = runs[1:]
    runs = runs + [(0, 10**6)]
    want = [int(c) for b in data for c in format(b, "08b")]
    out = []
    for i, bit in zip(range(0, 2 * len(want), 2), want):
        assert runs[i][0] == 1 and runs[i + 1][0] == 0
        out.append((runs[i][1] * ns, runs[i + 1][1] * ns, bit))
    return out


DATA = bytes([0b10110010, 0x00, 0xFF, 0x5A])


def test_default_timing_is_micropython_neopixel():
    rp2, driver = load()
    assert driver.DEFAULT_TIMING_NS == (400, 850, 800, 450)
    bits = waveform(rp2, driver, driver.DEFAULT_TIMING_NS, DATA)
    for hi, lo, bit in bits[:-1]:  # the last low runs into the idle line
        if bit:
            assert (hi, lo) == (800, 450)
        else:
            assert (hi, lo) == (400, 850)
    assert driver.cycles(driver.DEFAULT_TIMING_NS) == (8, 8, 9)


def test_configured_timing_is_honoured():
    rp2, driver = load()
    timing = (350, 900, 750, 500)
    for hi, lo, bit in waveform(rp2, driver, timing, DATA)[:-1]:
        assert (hi, lo) == ((750, 500) if bit else (350, 900))


def test_bad_timings_are_rejected():
    _, driver = load()
    for bad in ((400, 850, 800, 400), (400, 850, 400, 850), (2000, 0, 2000, 0), "fast", (400, 850)):
        with pytest.raises(driver.TimingError):
            driver.cycles(bad)
