"""Runs every registered effect under the MicroPython unix port and checks
that render() and the output stage do not allocate.

    micropython tools/mp_check.py [coords_file] [frames]

Run from the repository root. Prints one line per effect and exits 1 on any
failure. Uses only what MicroPython provides.
"""

import gc
import sys

sys.path.insert(0, ".")

import random

import effects
from effects import _coords, _output

N = 500


def measure(fn, *args):
    gc.collect()
    gc.disable()
    before = gc.mem_alloc()
    fn(*args)
    used = gc.mem_alloc() - before
    gc.enable()
    return used


def run_frames(effect, stage, raw, out, frames, step):
    t = 0
    for _ in range(frames):
        effect.render(t, raw)
        stage.process(raw, out, 200)
        t += step


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "coordinates/coords_compact.txt"
    frames = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    coords = _coords.load(path, N)
    failed = 0
    for name in effects.EFFECTS:
        try:
            mod = effects.load(name)
            ctx = effects.Ctx(N, coords, effects.defaults(mod), random)
            effect = mod.Effect(ctx)
            raw = bytearray(3 * N)
            out = bytearray(3 * N)
            stage = _output.Output(N)
            # Warm up: first frames may set up caches (param reads etc.).
            run_frames(effect, stage, raw, out, 3, 33)
            used = measure(run_frames, effect, stage, raw, out, frames, 33)
            ok = used == 0
            print("%-14s %s  %d frames allocated %d bytes" % (name, "ok  " if ok else "FAIL", frames, used))
            if not ok:
                failed += 1
        except Exception as e:
            failed += 1
            print("%-14s FAIL %s: %s" % (name, type(e).__name__, e))
            sys.print_exception(e)
    sys.exit(1 if failed else 0)


main()
