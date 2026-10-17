# Effect API

The same effect source runs unmodified on MicroPython (Pico 2W), CPython (headless renderer, tests, CI) and in the browser (Pyodide). Effects are pure: no hardware, network, sleeps, globals or I/O. The runtime owns the clock, frame rate, brightness, power limiting, colour order, the watchdog and networking.

This document is the contract between the effects in `effects/`, the firmware runtime, the headless renderer (`render.py`) and the browser viewer.

## Package layout

```
effects/
  __init__.py      # EFFECTS (ordered), load(name), defaults(module), Ctx
  _util.py         # shared helpers: fixed-point tables, Layout, Param, Phase
  _output.py       # the output stage (power cap, brightness, gamma, colour order)
  _coords.py       # coordinate loading and the legacy-coordinates adapter
  rainbow_cycle.py # one effect per module, imported lazily by name
  ...
```

- `EFFECTS` is ordered. It drives the Home Assistant effect list and the auto cycle. `"auto"` is a runtime mode, not an effect.
- Module name == `NAME` == the name in `EFFECTS`. The current names are the ones Home Assistant already knows (`rainbow_cycle`, `christmas`, ...), so existing automations keep working.
- One module per effect, so a single effect can be hot-pushed over OTA without a firmware rebuild.
- Modules whose names start with `_` are shared code, not effects.

## Effect module shape

```python
NAME = "rainbow_cycle"
DURATION_MS = 20000          # suggested time in the auto cycle; effects never "finish"
PARAMS = {                   # optional; exposed to HA, the renderer CLI and the viewer
    "speed": (1.0, 0.1, 5.0),  # (default, min, max)
}
NEEDS_COORDS = False         # True: the scheduler skips it when no calibration is loaded

class Effect:
    def __init__(self, ctx):
        # Precompute here; allocation and floats are fine. Keep it under ~1 s on the Pico.
        ...

    def render(self, t_ms, frame):
        # Fill frame (every pixel, every call). No allocation, no sleeps, return None.
        ...
```

### ctx

| Attribute | Meaning |
|---|---|
| `ctx.n` | LED count (500). |
| `ctx.coords` | `None`, or a list of `n` entries, each `(x, y, z)` or `None`; see [Coordinates](#coordinates). |
| `ctx.params` | Dict of current param values. Values may be replaced between frames, so read them in `render()` (see `Param` below). |
| `ctx.random` | Random source with `seed()`, `random()` and `getrandbits(k)`. On the Pico this is the `random` module; in the renderer and viewer it is `random.Random(seed)`. Use this, never the global `random`. |

`effects.Ctx(n, coords, params, random)` builds one; the runtime may use any object with these attributes.

### render(t_ms, frame)

- `t_ms`: int milliseconds since the effect started; monotonic, non-decreasing. The runtime restarts the effect (new `Effect(ctx)`, `t_ms` from 0) before `t_ms` reaches 2^29 (about 6 days), so it always stays a MicroPython small int.
- `frame`: `bytearray(3 * n)`, RGB order, linear 0-255, **not** brightness-scaled or power-capped. Write every pixel on every call; the buffer is reused and holds the previous frame.
- Frame rate is the runtime's choice (target 30-60 fps, measured) and may vary. Animate from `t_ms`, never from a frame count.

## Writing effects for MicroPython

`render()` must not allocate. On the Pico every float is a heap object, so one float operation per LED means thousands of allocations and frequent GC pauses per frame. In practice:

- Keep `render()` to integer arithmetic on small ints (< 2^30). Precompute per-LED values as ints in `__init__` and store them in `bytearray` or `array("H"/"h")`, whose items read back as small ints.
- Don't create tuples, lists, strings, closures or generators in `render()`. Unpacking an existing module-level tuple (`r, g, b = COLOR`) is fine.
- Call `ctx.random.getrandbits(k)` (k <= 30) rather than `random()`, which returns a float.
- Read float params through `_util.Param`, which converts on change only.
- No f-strings with format specs, walrus, `match`, dataclasses or typing at runtime. Standard modules other than `math` and `array` may not exist on the device.
- `micropython tools/mp_check.py` runs every effect under the MicroPython unix port and fails if `render()` plus the output stage allocates a single byte.

`effects/_util.py` provides fixed-point helpers. `ONE = 4096` means 1.0 for heights, radii, angles (a full turn) and phases:

| Helper | Use |
|---|---|
| `Layout(ctx)` | Per-LED integer tables: `known`, `h` (height 0..ONE), `theta` (azimuth 0..ONE-1, from `atan2(z, x)`), `r` (radius, ONE = tree's outer radius), `x`, `z`. Unknown LEDs get `known[i] = 0` and `h = i/n`, so string-order fallbacks keep working. |
| `Param(ctx, name, default, scale=ONE)` | `.get()` returns `int(value * scale)`, recomputed only when the value object changes. |
| `Phase(period_ms)` | `.update(t_ms, speed)` returns 0..ONE-1, one cycle per `period_ms` at `speed == ONE`; accumulates per-frame deltas, so speed changes stay smooth. |
| `WHEEL`, `SIN` | 256-entry colour wheel (RGB triples) and sine (0-255) tables. |

## Output stage

The runtime applies this to every frame, effect or calibration pattern, in this order:

```
raw RGB frame -> power cap (on the raw frame) -> brightness -> gamma (optional) -> colour order -> driver
```

`effects/_output.py` is the reference implementation, used unchanged by the renderer and viewer. The firmware should use it too, or a faster version that produces byte-identical output.

- **Power cap**: if the raw frame's channel sum exceeds the budget, scale every channel by `budget / sum`. The default budget is 13% of full white (`n * 765 * 13 // 100`), the original firmware's `MAX_POWER`.
- **Brightness**: 0-255, applied after the cap. Brightness 255 means "as bright as the budget allows" and lower values dim proportionally; previously the cap clipped dense effects above about 13-40% brightness (review H7).
- **Gamma**: off by default (linear, as before). `Output(n, gamma=2.2)` enables it.
- **Colour order**: `Output(n, order="RGB")` emits channels in the given order. The renderer and viewer use RGB. The firmware passes what its driver needs. Per review M8, the 12V WS2811 string is wired RGB, and the old code only worked because two GRB swaps cancelled out.
- `Output.process(src, dst, brightness)` is integer-only, does not allocate, and sets `raw_total`, `capped` and `factor` (cap factor in 1/4096) for telemetry.
- Current estimate: `_output.current_ma(channel_sum)` assumes 20 mA per channel at 255 (60 mA per pixel at full white). The default budget is about 3.9 A.

## Coordinates

Per-LED `(x, y, z)` or `None`, in this frame:

- y up: y = 0 at the lowest LED and y = 1 at the highest.
- x and z in the same units as y, so a cone with radius 0.4 x height has |xz| <= 0.4.
- Origin on the trunk axis.
- +x is the front camera's image right. Right-handed, so the front camera sits on +z looking towards -z.

`effects/_coords.py` loads:

- `coords.txt`: one `x,y,z` or `null` line per LED, already in this frame. The calibration pipeline will produce it.
- `coordinates/coords_compact.txt` (legacy): `adapt_legacy()` rescales it into the frame. It shifts y to start at 0 and divides by the height, centres x/z on the LED centroid, and multiplies x/z by 608/1080 to undo the anisotropic pixel normalisation in `analyze.py` (review H8). The result is a cone with an outer radius of about 0.3 x height, which is plausible for the 6 ft tree.

`_coords.load(path, n)` picks the adapter by file name and returns `None` when the file is missing. The renderer and viewer prefer `coordinates/coords.txt` and fall back to the legacy file.

## Determinism

Effects use `ctx.random` only, so `./render.py <effect> --seed N` is reproducible. `metrics.json` records a `frames_sha256` of all output frames for comparisons. MicroPython's `random` is a different generator from CPython's, so device output matches the renderer's in look, not byte-for-byte.

## Tooling

| Command | Purpose |
|---|---|
| `./render.py <effect>` | Headless render: `renders/<effect>/{anim.gif,sheet.png,metrics.json}`. Options: `--seconds --fps --seed --param k=v --brightness --budget-percent --gamma --view raw --mp4 --watch`. |
| `./render.py all` | Every effect, plus `renders/overview.png` and `renders/summary.json`. |
| `micropython tools/mp_check.py` | Allocation check under MicroPython. |

The renderer draws front, side and top projections of the real coordinates. Display brightness is gamma-lifted (1/2.2) so that dim output stays visible; the numbers in `metrics.json` are the actual output values.
