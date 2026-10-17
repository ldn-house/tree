#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Scaffold a new effect, register it, and render it once.

    ./new_effect.py aurora              # effects/aurora.py, appended to EFFECTS
    ./new_effect.py aurora --flat       # template that doesn't need coordinates

Then iterate with either:
    ./render.py aurora --watch          # re-renders renders/aurora/ on every save
    ./serve.py                          # browser viewer, hot-reloads on save
"""

import argparse
import keyword
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EFFECTS = ROOT / "effects"

TEMPLATE_3D = '''"""{title}: describe the look here."""

from effects._util import ONE, MASK, WHEEL, Layout, Param, Phase

NAME = "{name}"
DURATION_MS = 15000
PARAMS = {{
    "speed": (1.0, 0.1, 5.0),
    "bands": (1, 1, 6),
}}
NEEDS_COORDS = True

PERIOD_MS = 4000  # one cycle at speed 1


class Effect:
    def __init__(self, ctx):
        # Precompute per-LED tables here; floats and allocation are fine.
        self.n = ctx.n
        lay = Layout(ctx)
        self.known = lay.known
        self.h = lay.h  # height, 0..ONE
        self.theta = lay.theta  # angle round the trunk, 0..ONE-1
        self.speed = Param(ctx, "speed", 1.0)
        self.bands = Param(ctx, "bands", 1, 1)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        # Integer maths only: no floats, tuples or other allocation per frame.
        ph = self.phase.update(t_ms, self.speed.get())
        bands = self.bands.get()
        known = self.known
        h = self.h
        theta = self.theta
        for i in range(self.n):
            k = 3 * i
            if not known[i]:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            # Rainbow bands spiralling round the tree.
            p = (h[i] * bands + theta[i] + ph) & MASK
            o = (p >> 4) * 3
            frame[k] = WHEEL[o]
            frame[k + 1] = WHEEL[o + 1]
            frame[k + 2] = WHEEL[o + 2]
'''

TEMPLATE_FLAT = '''"""{title}: describe the look here."""

from effects._util import ONE, SIN, Param, Phase

NAME = "{name}"
DURATION_MS = 15000
PARAMS = {{
    "speed": (1.0, 0.1, 5.0),
}}
NEEDS_COORDS = False

PERIOD_MS = 3000  # one pulse at speed 1
COLOR = (255, 120, 20)


class Effect:
    def __init__(self, ctx):
        # Precompute per-LED tables here; floats and allocation are fine.
        self.n = ctx.n
        self.speed = Param(ctx, "speed", 1.0)
        self.phase = Phase(PERIOD_MS)

    def render(self, t_ms, frame):
        # Integer maths only: no floats, tuples or other allocation per frame.
        ph = self.phase.update(t_ms, self.speed.get())
        r, g, b = COLOR
        n = self.n
        for i in range(n):
            # A wave of brightness travelling along the string.
            v = SIN[((i * 256 // n) * 4 - (ph >> 4)) & 255]
            k = 3 * i
            frame[k] = r * v >> 8
            frame[k + 1] = g * v >> 8
            frame[k + 2] = b * v >> 8
'''


def register(name):
    init = EFFECTS / "__init__.py"
    src = init.read_text()
    m = re.search(r"EFFECTS = \[\n(.*?)\]", src, re.S)
    if not m:
        sys.exit("could not find the EFFECTS list in effects/__init__.py; add the name by hand")
    body = m.group(1)
    if not body.endswith("\n"):
        body += "\n"
    body += f'    "{name}",\n'
    init.write_text(src[: m.start(1)] + body + src[m.end(1):])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="module name: lowercase letters, digits and underscores")
    ap.add_argument("--flat", action="store_true", help="template that works without coordinates")
    ap.add_argument("--no-render", action="store_true", help="skip the first render")
    args = ap.parse_args()

    name = args.name
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name) or keyword.iskeyword(name):
        sys.exit(f"invalid effect name {name!r}: use lowercase letters, digits and underscores")
    path = EFFECTS / f"{name}.py"
    if path.exists():
        sys.exit(f"{path.relative_to(ROOT)} already exists")
    sys.path.insert(0, str(ROOT))
    import effects

    if name in effects.EFFECTS:
        sys.exit(f"{name!r} is already in EFFECTS")

    title = name.replace("_", " ").capitalize()
    template = TEMPLATE_FLAT if args.flat else TEMPLATE_3D
    path.write_text(template.format(name=name, title=title))
    register(name)
    print(f"created {path.relative_to(ROOT)} and added {name!r} to EFFECTS")
    if not args.no_render:
        r = subprocess.run([sys.executable, str(ROOT / "render.py"), name, "--seconds", "4"], cwd=ROOT)
        if r.returncode:
            return r.returncode
    print(f"\nnext: ./render.py {name} --watch   or   ./serve.py (viewer hot-reloads)")
    print(f"before a PR: ./smoke.py {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
