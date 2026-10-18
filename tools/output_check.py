"""Checks the output stage (effects/_output.py) under MicroPython or CPython.

    micropython tools/output_check.py      # from the repository root
    python3 tools/output_check.py

- parity: where the native (viper) kernels are available, they must produce
  exactly the same output bytes and dither remainders as the plain-Python
  ones, for every combination of dither, gamma, white balance, colour order,
  brightness and cap;
- allocation: process() must not allocate (MicroPython only);
- timing: ms per process() call for 500 LEDs, both kernels.

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
for dither in (True, False):
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
    for cfg in (CONFIGS[0], (True, 2.2, (1.0, 0.85, 0.7), "RGB"), (False, None, None, "RGB")):
        line = "timing %-48s python %.3f ms" % (repr(cfg), timing(py, cfg, raw))
        if native:
            line += "  native %.3f ms" % timing(native, cfg, raw)
        print(line)
    sys.exit(1 if failed else 0)


main()
