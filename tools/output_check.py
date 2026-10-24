"""Checks the output stage (effects/_output.py) under MicroPython or CPython.

    micropython tools/output_check.py      # from the repository root
    python3 tools/output_check.py

- parity: where the native (viper) kernels are available, they must produce
  exactly the same output bytes and dither state as the plain-Python ones,
  for every combination of dither mode (spatial, temporal, off), gamma,
  white balance, colour order, brightness and cap;
- allocation: process() must not allocate (MicroPython only);
- timing: ms per process() call for 500 LEDs, both kernels; and the
  colour floor (ColourFloor): ms to analyse one frame at once, and the
  heap it allocates doing so (MicroPython), for a solid warm white, a
  rainbow and noise at each accuracy.

Without native kernels, parity is reported as unverified; allocation and
fallback timing still run. Timings describe the current host, not the Pico.

Exits 1 on any failure.
"""

import gc
import sys
import time

sys.path.insert(0, ".")

from effects import _output

N = 500
MP = sys.implementation.name == "micropython"


def ticks_us():
    if MP:
        return time.ticks_us()
    return int(time.perf_counter() * 1000000)


def frames(count, seed):
    """Deterministic pseudo-random raw frames (dense and sparse)."""
    out = []
    x = seed
    for f in range(count):
        b = bytearray(3 * N)
        sparse = f % 3 == 2
        for k in range(3 * N):
            x = (x * 1103515245 + 12345) & 0x7FFFFFFF
            v = (x >> 16) & 255
            b[k] = (v if v > 200 else 0) if sparse else v
        out.append(b)
    return out


CONFIGS = []
for dither in ("spatial", "temporal", False):
    for gamma in (None, 2.2):
        for white in (None, (1.0, 0.85, 0.7)):
            for order in ("RGB", "GRB"):
                CONFIGS.append((dither, gamma, white, order))


def run(cfg, kernels, raws, brightnesses):
    dither, gamma, white, order = cfg
    o = _output.Output(N, gamma=gamma, order=order, dither=dither, white=white)
    saved = (_output._lut, _output._apply)
    _output._lut, _output._apply = kernels
    try:
        out = []
        dst = bytearray(3 * N)
        for raw in raws:
            for b in brightnesses:
                o.process(raw, dst, b)
                out.append(bytes(dst))
        out.append(bytes(o.res))
        return out
    finally:
        _output._lut, _output._apply = saved


def measure_alloc(o, raw, dst):
    o.process(raw, dst, 128)
    gc.collect()
    gc.disable()
    before = gc.mem_alloc()
    for b in (3, 40, 128, 255):
        o.process(raw, dst, b)
    used = gc.mem_alloc() - before
    gc.enable()
    return used


def timing(kernels, cfg, raw):
    dither, gamma, white, order = cfg
    o = _output.Output(N, gamma=gamma, order=order, dither=dither, white=white)
    dst = bytearray(3 * N)
    saved = (_output._lut, _output._apply)
    _output._lut, _output._apply = kernels
    try:
        o.process(raw, dst, 200)
        t = ticks_us()
        for _ in range(50):
            o.process(raw, dst, 200)
        return (ticks_us() - t) / 50000
    finally:
        _output._lut, _output._apply = saved


def main():
    failed = 0
    py = (_output._lut_py, _output._apply_py)
    native = (_output._lut, _output._apply) if _output.NATIVE else None
    print("implementation: %s; native kernels: %s" % (sys.implementation.name, "yes" if native else "no"))
    raws = frames(6, 1)
    brightnesses = (1, 4, 9, 33, 128, 255)
    if native:
        for cfg in CONFIGS:
            a = run(cfg, py, raws, brightnesses)
            b = run(cfg, native, raws, brightnesses)
            ok = a == b
            failed += not ok
            print("parity %-48s %s" % (repr(cfg), "ok" if ok else "MISMATCH"))
    else:
        print("native parity: NOT VERIFIED (native kernels unavailable)")
    if MP:
        raw = raws[0]
        dst = bytearray(3 * N)
        for cfg in CONFIGS:
            dither, gamma, white, order = cfg
            o = _output.Output(N, gamma=gamma, order=order, dither=dither, white=white)
            used = measure_alloc(o, raw, dst)
            failed += used != 0
            print("alloc  %-48s %d bytes%s" % (repr(cfg), used, "" if used == 0 else "  FAIL"))
    raw = raws[0]
    for cfg in (CONFIGS[0], ("spatial", 2.2, (1.0, 0.85, 0.7), "RGB"), ("temporal", None, None, "RGB"),
                (False, None, None, "RGB")):
        line = "timing %-48s python %.3f ms" % (repr(cfg), timing(py, cfg, raw))
        if native:
            line += "  native %.3f ms" % timing(native, cfg, raw)
        print(line)
    for name, raw in colour_frames():
        for acc in (0.02, 0.01, 0.005):
            ms, alloc, pct = colour_timing(raw, acc)
            print("colour %-10s accuracy %.3f  %.2f ms  %s  floor %.2f%%" % (
                name, acc, ms, "%d bytes allocated" % alloc if MP else "", pct))
    sys.exit(1 if failed else 0)


def colour_frames():
    """Solid warm white (one colour), a rainbow along the string and noise
    (every analysed pixel different)."""
    rainbow = bytearray(3 * N)
    for i in range(N):
        h = i * 768 // N
        seg, x = h // 256, h % 256
        c = ((255 - x, x, 0), (0, 255 - x, x), (x, 0, 255 - x))[seg]
        rainbow[3 * i:3 * i + 3] = bytes(c)
    return [("warm", bytearray((255, 87, 14) * N)), ("rainbow", rainbow), ("noise", frames(1, 7)[0])]


def colour_timing(raw, accuracy):
    best = None
    alloc = 0
    for _ in range(3):
        cf = _output.ColourFloor(_output.Output(N), accuracy)
        gc.collect()
        before = gc.mem_alloc() if MP else 0
        t = ticks_us()
        cf.add(raw)
        us = ticks_us() - t
        if MP:
            alloc = gc.mem_alloc() - before
        best = us if best is None else min(best, us)
    return best / 1000, alloc, cf.pct


main()
