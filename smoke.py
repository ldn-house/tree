#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Smoke-test every registered effect. Exits non-zero on any failure.

Checks, per effect:
- module metadata: NAME matches, DURATION_MS > 0, PARAMS are (default, min, max)
- renders N seconds under CPython with real coordinates: no exceptions, not
  all black, estimated current never above the cap (amps)
- deterministic: the same seed renders byte-identical frames twice
- survives every param at its min and its max, and having no coordinates
- compiles with mpy-cross (via `uvx --from mpy-cross`, when uv is available)
- under the MicroPython unix port (when `micropython` is on PATH, or given with
  --micropython): imports, renders and allocates nothing per frame

Usage:
    ./smoke.py                     # everything available locally
    ./smoke.py rainbow_cycle       # just one effect
    ./smoke.py --require-micropython --require-mpy-cross   # CI: missing tools fail
"""

import argparse
import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import effects  # noqa: E402
import render  # noqa: E402
from effects import _output  # noqa: E402


class Report:
    def __init__(self):
        self.failures = []

    def check(self, ok, name, what):
        if not ok:
            self.failures.append(f"{name}: {what}")
            print(f"  FAIL {what}")
        return ok


def check_metadata(rep, name, mod):
    rep.check(getattr(mod, "NAME", None) == name, name, f"NAME is {getattr(mod, 'NAME', None)!r}, expected {name!r}")
    d = getattr(mod, "DURATION_MS", None)
    rep.check(isinstance(d, int) and d > 0, name, f"DURATION_MS must be a positive int, got {d!r}")
    rep.check(isinstance(getattr(mod, "NEEDS_COORDS", None), bool), name, "NEEDS_COORDS must be a bool")
    rep.check(hasattr(mod, "Effect"), name, "no Effect class")
    for k, spec in getattr(mod, "PARAMS", {}).items():
        ok = (isinstance(spec, tuple) and len(spec) == 3
              and all(isinstance(v, (int, float)) for v in spec) and spec[1] <= spec[0] <= spec[2])
        rep.check(ok, name, f"PARAMS[{k!r}] must be (default, min, max) with min <= default <= max, got {spec!r}")


# Effects are pure: no hardware, I/O, clocks or unseeded randomness.
FORBIDDEN_IMPORTS = {"machine", "network", "neopixel", "rp2", "time", "utime", "random", "urandom",
                     "os", "uos", "socket", "_thread", "asyncio", "uasyncio", "gc", "sys"}


def check_imports(rep, name):
    tree = ast.parse((ROOT / "effects" / f"{name}.py").read_text())
    for node in ast.walk(tree):
        mods = []
        if isinstance(node, ast.Import):
            mods = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            mods = [node.module]
        for m in mods:
            rep.check(m.split(".")[0] not in FORBIDDEN_IMPORTS, name,
                      f"imports {m} (line {node.lineno}); effects must be pure, use ctx.random for randomness")


def check_power(rep, name, m, label=""):
    """The output stage must hold the current cap: on average over the run,
    and within 1 LSB per lit channel on any single frame (the most temporal
    dithering can add)."""
    p = m["power"]
    peak = p["current_a"]["peak"]
    suffix = f" with {label}" if label else ""
    n = m["led_count"]
    one_lsb_a = _output.current_a(3 * n, p["ma_per_channel"])
    rep.check(peak is None or peak <= p["limit_a"] + one_lsb_a, name,
              f"peak current {peak} A above the {p['limit_a']} A cap (+1 LSB per channel){suffix}")
    rep.check(p["over_budget_frames"] == 0, name,
              f"{p['over_budget_frames']} frames more than 1 LSB per lit channel above the "
              f"{p['budget_channel_sum']} channel-sum budget{suffix}")
    # Over a run, dithering's remainders can add at most 1 per channel in total.
    slack = 3 * n / max(1, m["frames"]) + 1
    mean_over = p.get("mean_over_budget")
    rep.check(mean_over is None or mean_over <= slack, name,
              f"mean output {mean_over} above the channel-sum budget{suffix}")


def check_render(rep, name, mod, coords, seconds, fps):
    res = render.run_effect(name, seconds, fps, seed=1, coords=coords)
    m = res["metrics"]
    for e in m["exceptions"]:
        print(e["error"])
    rep.check(not m["exceptions"], name, f"exception at frame {m['exceptions'][0]['frame'] if m['exceptions'] else ''}")
    rep.check(m["frames"] == m["frames_requested"], name, f"rendered {m['frames']}/{m['frames_requested']} frames")
    rep.check(not m["all_black"], name, "output is black for the whole run")
    check_power(rep, name, m)
    again = render.run_effect(name, seconds, fps, seed=1, coords=coords)["metrics"]
    rep.check(again["frames_sha256"] == m["frames_sha256"], name, "not deterministic for a fixed seed")
    p = m["power"]
    if not m["frames"]:
        return
    print(f"       {m['frames']} frames, cap {100 * p['cap_engaged_rate']:.0f}%, "
          f"peak {p['current_a']['peak']:.2f} A, lit {m['lit_fraction']['mean']:.0%}, "
          f"{res['perf']['render_ms_mean']} ms/frame")


def check_variants(rep, name, mod, coords, fps):
    spec = getattr(mod, "PARAMS", {})
    variants = [("no coordinates", {}, None)]
    if spec:
        variants += [("params at min", {k: v[1] for k, v in spec.items()}, coords),
                     ("params at max", {k: v[2] for k, v in spec.items()}, coords)]
    for label, over, c in variants:
        m = render.run_effect(name, 2.0, fps, seed=2, overrides=over, coords=c)["metrics"]
        if m["exceptions"]:
            print(m["exceptions"][0]["error"])
        rep.check(not m["exceptions"], name, f"exception with {label}")
        check_power(rep, name, m, label)
        if label == "no coordinates" and not mod.NEEDS_COORDS:
            rep.check(not m["all_black"], name, "black without coordinates but NEEDS_COORDS is False")


def check_mpy_cross(rep, required):
    uvx = shutil.which("uvx")
    files = sorted(str(p.relative_to(ROOT)) for p in (ROOT / "effects").glob("*.py"))
    if not uvx:
        rep.check(not required, "mpy-cross", "uvx not found")
        print("mpy-cross: skipped (uvx not found)")
        return
    print(f"mpy-cross: compiling {len(files)} files")
    for f in files:
        r = subprocess.run([uvx, "--from", "mpy-cross", "mpy-cross", "-o", os.devnull, f],
                           cwd=ROOT, capture_output=True, text=True)
        if r.returncode:
            print(r.stdout + r.stderr)
        rep.check(r.returncode == 0, "mpy-cross", f"{f} does not compile")


def check_micropython(rep, exe, required, only):
    exe = exe or shutil.which("micropython")
    if not exe:
        rep.check(not required, "micropython", "micropython not found")
        print("micropython: skipped (not on PATH; see .github/workflows/effects.yml to build the unix port)")
        return
    print(f"micropython: {exe} tools/mp_check.py")
    r = subprocess.run([exe, "tools/mp_check.py", "coordinates/coords_compact.txt", "60"],
                       cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if not only or line.split(" ")[0] in only or not line[:1].isalpha():
            print("  " + line)
    rep.check(r.returncode == 0, "micropython", "allocation/compat check failed")
    print(f"micropython: {exe} tools/output_check.py")
    r = subprocess.run([exe, "tools/output_check.py"], cwd=ROOT, capture_output=True, text=True)
    out = r.stdout + r.stderr
    for line in out.splitlines():
        if not line.endswith(" ok") and not line.endswith(" 0 bytes"):
            print("  " + line)
    rep.check(r.returncode == 0, "micropython", "output stage parity/allocation check failed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("effects", nargs="*", help="effects to check (default: all registered)")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--micropython", help="path to the MicroPython unix port binary")
    ap.add_argument("--require-micropython", action="store_true")
    ap.add_argument("--require-mpy-cross", action="store_true")
    args = ap.parse_args()

    rep = Report()
    names = args.effects or effects.EFFECTS
    on_disk = sorted(p.stem for p in (ROOT / "effects").glob("*.py") if not p.stem.startswith("_"))
    disabled = list(getattr(effects, "DISABLED", []))
    unregistered = [n for n in on_disk if n not in effects.EFFECTS and n not in disabled]
    rep.check(not unregistered, "effects", f"modules in neither EFFECTS nor DISABLED: {', '.join(unregistered)}")
    both = [n for n in disabled if n in effects.EFFECTS]
    rep.check(not both, "effects", f"in both EFFECTS and DISABLED: {', '.join(both)}")
    missing = [n for n in effects.EFFECTS + disabled if n not in on_disk]
    rep.check(not missing, "effects", f"listed but no module file: {', '.join(missing)}")
    rep.check(len(set(effects.EFFECTS)) == len(effects.EFFECTS), "effects", "duplicate names in EFFECTS")
    coords, src = render.find_coords()
    print(f"coords: {src}")
    for name in names:
        print(name)
        try:
            mod = effects.load(name)
        except Exception as e:  # noqa: BLE001
            rep.check(False, name, f"import failed: {e!r}")
            continue
        check_metadata(rep, name, mod)
        check_imports(rep, name)
        check_render(rep, name, mod, coords, args.seconds, args.fps)
        check_variants(rep, name, mod, coords, args.fps)
    check_mpy_cross(rep, args.require_mpy_cross)
    check_micropython(rep, args.micropython, args.require_micropython, set(args.effects))

    if rep.failures:
        print(f"\n{len(rep.failures)} failure(s):")
        for f in rep.failures:
            print(f"  {f}")
        return 1
    print(f"\nall {len(names)} effect(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
