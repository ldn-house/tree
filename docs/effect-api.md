# Effect API

The same effect source runs unmodified on MicroPython (Pico 2W), CPython (headless renderer, tests, CI) and in the browser (Pyodide). Effects are pure: no hardware, network, sleeps, globals or I/O. The runtime owns the clock, frame rate, brightness, power limiting, colour order, the watchdog and networking.

This document is the contract between the effects in `effects/`, the firmware runtime, the headless renderer (`render.py`) and the browser viewer.

## Package layout

```
effects/
  __init__.py      # EFFECTS (ordered), load(name), defaults(module), Ctx
  _util.py         # shared helpers: fixed-point tables, Layout, Param, Phase
  _output.py       # the output stage (power cap, brightness and its floor, gamma, colour order)
  _coords.py       # coordinate loading and the legacy-coordinates adapter
  rainbow_cycle.py # one effect per module, imported lazily by name
  ...
```

- `EFFECTS` is ordered. It drives the Home Assistant effect list and the auto cycle. `"auto"` is a runtime mode, not an effect.
- An effect with `AUTO = False` is registered (Home Assistant can select it, and takeovers can show it) but the auto cycle skips it: `solid` and the `f1_*` effects. If every eligible effect is `AUTO = False`, auto mode shows the fallback.
- `DISABLED` lists effect modules kept in the package but not registered (currently `sparkle`). Only `EFFECTS` may be exposed to Home Assistant, the auto cycle or tools; nothing may discover effects by listing `effects/*.py`. To re-enable an effect, move its name from `DISABLED` back into `EFFECTS`.
- Module name == `NAME` == the name in `EFFECTS`. The current names are the ones Home Assistant already knows (`rainbow_cycle`, `christmas`, ...), so existing automations keep working.
- One module per effect, so a single effect can be hot-pushed over OTA without a firmware rebuild.
- Modules whose names start with `_` are shared code, not effects.

## Effect module shape

```python
NAME = "rainbow_cycle"
DURATION_MS = 20000          # suggested time in the auto cycle; effects never "finish"
PARAMS = {                   # optional; exposed to HA, the renderer CLI and the viewer
    "speed": (1.0, 0.1, 5.0),  # (default, min, max)
    "color": rgb_spec(255, 0, 0),  # a colour: (0xFF0000, 0, 0xFFFFFF, "rgb")
}
NEEDS_COORDS = False         # True: the scheduler skips it when no calibration is loaded
AUTO = True                  # optional; False: never auto-cycled, only selected or taken over

class Effect:
    def __init__(self, ctx):
        # Precompute here; allocation and floats are fine. Keep it under ~1 s on the Pico.
        ...

    def render(self, t_ms, frame):
        # Fill frame (every pixel, every call). No allocation, no sleeps, return None.
        ...
```

### Colour params

A param whose spec is `(0xRRGGBB, 0, 0xFFFFFF, "rgb")` is a colour; build it with `_util.rgb_spec(r, g, b)`. Its value is a packed `0xRRGGBB` int, so reading it never allocates, and a tool that only knows `(default, min, max)` still sees a valid range.

- In `render()`, read it through `_util.ColorParam(ctx, name, default)`: `update()` returns True when the value changed (and on the first call) and sets `.r`, `.g`, `.b`. Rebuild anything derived from the colour only then.
- Values come in as `"#RRGGBB"`, `"r,g,b"`, `[r, g, b]` or the int (`_util.parse_rgb`); the runtime, `render.py --param color=#ff8000`, the viewer (a colour picker) and the firmware all store the int.
- Home Assistant shows each colour param as a `text` entity that takes `#RRGGBB`. A param named `color` is the effect's main colour: picking a colour on the HA light sets it (firmware `app/ha.py`).
- `GET /effects` marks colour params with `"type": "rgb"`.

### ctx

| Attribute | Meaning |
|---|---|
| `ctx.n` | LED count (500). |
| `ctx.coords` | `None`, or a list of `n` entries, each `(x, y, z)` or `None`; see [Coordinates](#coordinates). To change coordinates, the runtime passes a new list rather than editing this one, because derived tables are cached per list. |
| `ctx.params` | Dict of current param values. The runtime assigns new values into this same dict between frames, and never swaps the dict itself, since effects keep a reference to it. Read params in `render()` (see `Param` below). |
| `ctx.random` | Random source with `seed()`, `random()` and `getrandbits(k)`. On the Pico this is the `random` module; in the renderer and viewer it is `random.Random(seed)`. Use this, never the global `random`. |

`effects.Ctx(n, coords, params, random)` builds one; the runtime may use any object with these attributes.

### render(t_ms, frame)

- `t_ms`: int milliseconds since the effect started; monotonic, non-decreasing. A global speed control scales the clock: the runtime accumulates scaled time, so `t_ms` never jumps backwards. Helpers treat a single step over 1 s as a stall and don't animate across it. The runtime restarts the effect (new `Effect(ctx)`, `t_ms` from 0) before `t_ms` reaches 2^29 (about 6 days), so it always stays a MicroPython small int.
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
| `Layout(ctx)` | Per-LED integer tables: `known`, `h` (height 0..ONE), `theta` (azimuth 0..ONE-1, from `atan2(z, x)`), `r` (radius, ONE = tree's outer radius), `x`, `z`. Unknown LEDs get `known[i] = 0` and `h = i/n`, so string-order fallbacks keep working. The tables are built once per coordinates list and shared by every effect, so treat them as read-only. Building them is the expensive part of an effect's `__init__` (float maths per LED); sharing them keeps effect switches in the auto cycle from churning the heap. |
| `Param(ctx, name, default, scale=ONE)` | `.get()` returns `int(value * scale)`, recomputed only when the value object changes. |
| `Phase(period_ms)` | `.update(t_ms, speed)` returns 0..ONE-1, one cycle per `period_ms` at `speed == ONE`; accumulates per-frame deltas, so speed changes stay smooth. |
| `WHEEL`, `SIN` | 256-entry colour wheel (RGB triples) and sine (0-255) tables. |
| `rgb_spec(r, g, b)`, `ColorParam(ctx, name, default)` | A colour param's spec, and its allocation-free reader (`.update()`, `.r`, `.g`, `.b`). See [Colour params](#colour-params). |

## Output stage

The runtime applies this to every frame, effect or calibration pattern, in this order:

```
raw RGB frame -> power cap (on the raw frame) -> brightness (above the floor)
  -> white balance (optional) -> gamma (optional) -> rounding / dithering -> colour order -> driver
```

`effects/_output.py` is the reference implementation, used unchanged by the renderer, the viewer and the firmware (which also vendors a byte-identical copy as `firmware/app/_output_fallback.py`).

`Output(n, budget=None, gamma=None, order="RGB", dither="spatial", white=None, floor=FLOOR_PCT, dither_min=DITHER_MIN)`:

- **Power cap**: a current limit. If the raw frame's channel sum exceeds the budget, every channel is scaled by `budget / sum`. The strip is 12 V WS2811 bullets on a single feed, which binds at about 1.5 A, so the default is `LIMIT_A = 1.5`. `budget_for_current(limit_a, ma_per_channel)` converts amps to the channel-sum budget `Output(budget=...)` takes: 47,812 at the defaults. `Output(n)` with no budget uses that. `budget_for(n, percent)`, the original firmware's 13% of full white, remains for compatibility. The cap holds on every frame: when rounding or dithering could take a frame over the budget (only near the cap at high brightness), the excess is trimmed off lit channels in rotation.
- **Brightness**: 0-255, applied after the cap. Brightness 255 means "as bright as the budget allows" and lower values dim proportionally (review H7).
- **Brightness floor** (`floor=`, percent of full brightness, default `FLOOR_PCT = 4`; `set_floor()` changes it, `level(brightness)` returns the effective level in 1/256 units): below a few levels no rounding can show a colour. Brightness 3..255 (Home Assistant's 1..100%) maps linearly onto floor..100%; 1 and 2, which HA's slider can't pick, also give the floor, and 0 stays dark. 4% is the lowest level at which warm white (tone 2) keeps its hue: (3, 1, 0) on every LED under the 1.5 A cap (at 3% it would be (2, 1, 0), too yellow). `floor=0` is exactly the unfloored stage. Calibration frames use `floor=0` (and brightness 255, which no floor changes). In the firmware the floor is a persisted state setting (`floor` in `POST /state`, HA's "Brightness floor" number), not a config key, so it works on any recovery layer.
- **No dithering at a few levels** (`dither_min=`, default `DITHER_MIN = 9`): a channel whose target is below 9 levels is rounded to nearest on every LED, so a solid colour is one colour on the whole tree. On the tree a one-level split in a small channel shows as two tints (blue 0 next to blue 1 reads as yellow next to white; green 3 next to 4 at red 7 was visible too), while splits where the step is at most about 1/9 of the channel didn't show (red 9/10, green 4/5 at red 10, blue 1/2 at red 28). Nearest rounding is the closest whole-number colour to the target, which keeps the hue as close as integers allow, but the low end moves in whole levels: neighbouring slider steps can give the same colour. It's applied in the per-frame lookup table (a level below `dither_min` is rounded to a whole level there, which every rounding threshold then leaves alone), so the kernels and their cost are unchanged. `dither_min=0` dithers at every level.
- **Spatial dithering** (`dither="spatial"`, the default; `True` means the same; firmware `LED_DITHER`), from `dither_min` up: each LED has a fixed rounding threshold, shared by its three channels and spread evenly along the string by a low-discrepancy sequence (an additive recurrence with a golden-ratio step). A target of 9.6 is therefore sent as 10 on about 60% of LEDs and as 9 on the rest. Every LED stays within 1 of its exact value, the tree-wide average matches the target, and a static frame produces identical bytes on every frame, so nothing flickers.
- **Temporal dithering** (`dither="temporal"`, opt-in): carries each channel's remainder to the next frame instead, so every LED averages exactly over time. At the tree's 35-40 fps this flickers visibly at low levels (seen on the tree), so it is off by default.
- **No dithering** (`dither=False`): rounds to nearest. Calibration frames use this, for steady camera levels.
- **White balance** (`white=(r, g, b)`, firmware `WHITE_BALANCE`, default off): per-source-channel scale, each 0..1, applied with brightness. It only ever lowers channels, so the cap still holds.
- **Gamma**: off by default (linear). `Output(n, gamma=2.2)` applies `out = 255 * (lin / 255) ** gamma` through a 2041-entry table (1/8-level input steps); dithering keeps the low end usable. Gamma >= 1 never raises a channel above its linear value.
- **Colour order**: `Output(n, order="RGB")` emits channels in the given order. The renderer and viewer use RGB. Per review M8, the 12V WS2811 string is wired RGB.
- `Output.process(src, dst, brightness)` does not allocate. It sets `raw_total`, `capped` and `factor` (cap factor in 1/4096) for telemetry.
- **Speed**: on MicroPython the per-frame loops are viper code compiled at import from a source string (mpy-cross can't emit native code without `-march`). Elsewhere, or if that fails, plain-Python versions run; `_output.NATIVE` says which. Both produce identical bytes, which `tools/output_check.py` checks under MicroPython. On the native 32-bit MicroPython 1.27 build the stage costs about 0.3 ms per frame (the Python fallback about 1.9 ms; the old stage 1.2 ms).
- **Current estimate**: `_output.current_a(channel_sum, ma_per_channel)` assumes `MA_PER_CHANNEL = 8` mA per channel at 255 for a 12 V WS2811 bullet. Estimate a frame's current from the bytes actually sent, `current_a(sum(dst))`, as the firmware, renderer and viewer do; it accounts for gamma, white balance and dithering.

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
| `./render.py <effect>` | Headless render: `renders/<effect>/{anim.gif,sheet.png,metrics.json}`. Options: `--seconds --fps --seed --param k=v --brightness --floor --limit-a --ma-per-channel --gamma --dither spatial|temporal|none --white R,G,B --view raw --mp4 --watch`. `metrics.json` reports the cap (`limit_a`, `budget_channel_sum`, `capped`, `cap_engaged_rate`) and estimated current in amps (`current_a` mean and peak, `uncapped_peak_a`), plus `per_frame.current_a` and `per_frame.capped`. |
| `./render.py all` | Every effect, plus `renders/overview.png` and `renders/summary.json`. |
| `micropython tools/mp_check.py` | Allocation check under MicroPython. |
| `micropython tools/output_check.py` | Output stage: native vs Python parity, allocation, timing. |
| `uv run --with pytest pytest tests/effects` | Output stage behaviour (the brightness floor, no split below `DITHER_MIN`, cap, rounding, dithering). |
| `./new_effect.py <name> [--flat]` | Scaffold `effects/<name>.py` from a template, register it in `EFFECTS` and render it once. |
| `./render.py <effect> --watch` | Re-render on every save in `effects/`. |
| `./serve.py` | Browser viewer (`viewer/`), hot-reloads changed effect files. |
| `./push_effect.py <effect...> --host H [--activate]` | Run the local checks, then hot-push those effects to the tree over Wi-Fi (see below). |
| `./smoke.py [effect...]` | Everything CI checks: metadata, pure imports, renders without exceptions, not all-black, within budget, deterministic, survives params at min/max and no coordinates, mpy-cross compile, MicroPython allocation check. |

The renderer draws front, side and top projections of the real coordinates. Display brightness is gamma-lifted (1/2.2) so that dim output stays visible; the numbers in `metrics.json` are the actual output values.

## Pushing one effect over OTA

`./push_effect.py <effect...> --host H [--activate]` hot-pushes effect modules to a running tree, with no firmware build and no reboot:

1. It runs the smoke checks on each effect and compiles it with mpy-cross. Failures stop the push.
2. It uploads with `PUT /effects/<name>`, sending `X-Content-SHA256`. With `--activate`, the last effect is pushed with `?activate=1`.
3. It re-reads `GET /effects` and checks that each effect is listed with `overlay: true` and no `error`.

Other options:
- `--delete` removes a pushed overlay, reverting to the firmware's copy.
- `--mpy` pushes mpy-cross bytecode. The `.mpy` version must match the device's MicroPython.

Credentials are read the way `flash.py` reads them:
- **Token**: from `--token`, `$TREE_TOKEN` or `HTTP_TOKEN` in `config.py`, sent as `Authorization: Bearer`.
- **Signature**: when `config.py` sets `UPDATE_KEY`, every push carries `X-Signature`, the hex HMAC-SHA256 keyed with `UPDATE_KEY` of `b"tree-effect\n" + name + b"\n" + body`. The name is included, so a signed effect can't be replayed under another name.
- **Port**: from `HTTP_PORT` or `OTA_PORT` in `config.py`, else 8080.

`--config PATH` reads another `config.py`. `tools/mock_tree.py [--token T] [--update-key K]` mirrors the endpoints for local testing.

The firmware (`firmware/app/api.py`) defines the device side:

| Endpoint | Behaviour |
|---|---|
| `GET /effects` | `{"effects": [{"name", "duration_ms", "needs_coords", "auto", "params": {k: {"default", "min", "max", "value", "type"?}}, "overlay", "eligible", "error"}], "running", "fx_error"}` (`type` is `"rgb"` for a colour) |
| `PUT` or `POST /effects/<name>` | Auth required (401). If the device has `UPDATE_KEY`, `X-Signature` must verify before any of the code runs (403). The body is module source, or `.mpy` (sniffed, or `?format=mpy`), at most 32 KB. Optional `X-Content-SHA256`; `?activate=1` also selects the effect. The device compile-checks the module, imports it under a staging name, builds `Effect(ctx)` and renders t=0 and t=33. Only then does it replace the overlay (`/fx/fx_<name>.py`) and hot-reload. Replies `200 {"effect", "overlay": true, "activated", "duration_ms", "params", "needs_coords"}`, or `400 {"error"}` and keeps the previous version. |
| `DELETE /effects/<name>` | Auth required. Removes the overlay. |

Limits:
- Only effect modules can be pushed. Shared modules (`_util`, `_output`, `_coords`) and `__init__` ship with firmware.
- A pushed effect with a new name is added without editing `EFFECTS`.
- Installing new firmware wipes all overlays.
