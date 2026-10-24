# Effect API

The same effect source runs unmodified on MicroPython (Pico 2W), CPython (headless renderer, tests, CI) and in the browser (Pyodide). Effects are pure: no hardware, network, sleeps, globals or I/O. The runtime owns the clock, frame rate, brightness, power limiting, colour order, the watchdog and networking.

This document is the contract between the effects in `effects/`, the firmware runtime, the headless renderer (`render.py`) and the browser viewer.

## Package layout

```
effects/
  __init__.py      # EFFECTS (ordered), load(name), defaults(module), Ctx
  _util.py         # shared helpers: fixed-point tables, Layout, Param, Phase
  _output.py       # the output stage (power cap, brightness and its floors, gamma, colour order)
  _coords.py       # coordinate loading and the legacy-coordinates adapter
  rainbow_cycle.py # one effect per module, imported lazily by name
  ...
```

- `EFFECTS` is ordered. It drives the Home Assistant effect list and the auto cycle. `"auto"` is a runtime mode, not an effect.
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
- **Colour floor** (`ColourFloor(stage, accuracy=COLOUR_ACCURACY)`): how many levels a colour needs depends on the colour. Pure red or green is itself at any level, but warm white (tone 2) at red 3-8 has no blue and reads as yellow; only from about red 11, (11, 4, 1), is it warm white again. The colour floor is the lowest level from which every step up the slider keeps an effect's colours within `accuracy`, and the stage uses the higher of it and `floor` (`set_colour_floor()`; `floor_pct()` returns the result, `colour_floor` the colour floor in percent):
  - Drift is the CIE 1976 u'v' distance between a pixel's wire colour and its exact target, treating wire values as linear light (the WS2811 dims by PWM) with sRGB primaries (`drift()`). On the tree, (3, 1, 0) for tone 2 is 0.012 and looked yellow, (22, 8, 1) is 0.005 and looked warm white. Every blue-less step of tone 2 is about 0.012-0.016, so the default `COLOUR_ACCURACY = 0.01` means "never the yellow". Settings are 0 (off) or 0.005-0.05 (`colour_accuracy()`). Output that rounds to all-dark isn't drift (that's what `floor` is for), so single-channel colours never hold the floor up.
  - Saturated colours: for a target further from white than `COLOUR_HUE_FROM = 0.105` (just past tone 2's 0.104), the allowed drift is `accuracy` times its distance over 0.105 (`tolerance()`), a hue angle of about 5.5° around white at 0.01. The same u'v' step is a much smaller change of hue for a saturated colour, and a fixed distance held rainbow_cycle at 35% on the tree, visibly too bright for HA's 1%. warm_white tones 0-2 keep the fixed threshold, and the more orange tone 3 (0.122) gets 16% more. On the tree, tone 0 at (5, 3, 1) (0.006) looked right, (4, 2, 1) and (6, 4, 1) (0.011 and 0.015) plain white, and (3, 1, 1) (0.045) pink.
  - The wire colour is this stage's: channels below `dither_min` are rounded to a whole level, and dithered channels count at their exact value. The colour only changes where a channel crosses a rounding boundary, so the search checks the ends of each interval between boundaries, top down, and is exact; `tests/effects` checks it against every scale `process()` can use. Levels are checked up to `COLOUR_FLOOR_MAX = 50`% (a drift higher up can't be fixed by a floor), and the colour floor never goes above that.
  - Mixed frames: pixels below half the frame's brightest are skipped (fades and tails, which would need several times the level), up to `COLOUR_SAMPLES = 32` of the rest are analysed, evenly spread along the string (identical colours once), and the colour floor is the level that `COLOUR_PCT = 90`% of them need.
  - Animation: frames are analysed at `COLOUR_AT_MS` (0, 1, 3 and 6 s) after `reset()`, pooling the pixels. Call `reset()` when the effect, its params, the coordinates or the accuracy change; each sample sets the floor when it is done, and the previous floor stays until the first one is. `add(raw)` analyses a frame at once; `start(raw)` copies it, and `step(pixels, us)` goes on with it until either limit is reached (each call makes some progress, and one colour can't be split). Nothing runs per frame otherwise, and the drift is computed in small integers (floats allocate on MicroPython), with floats only within 6% of the threshold.
  - Cost (MicroPython 1.27 unix build on a laptop, `tools/output_check.py`): 0.1 ms for a solid colour, about 1 ms for a rainbow or noise (32 different colours), allocating at most 70 KB. On the Pico one colour takes about 10 ms, and a rainbow analysed at once held a frame for 320 ms, so the firmware spreads each sample over the following frames, at most `COLOUR_US = 4000` µs per frame beyond one colour; `GET /` → `brightness_floor` reports the time (`colour_ms` for the last sample, `colour_call_ms_max` for the longest frame).
  - Floors at the default 1.5 A cap: warm_white tones 0-3 get 15.5, 16.2, 15.8 and 22.9% at 0.01 (8.3, 6.1, 11.0 and 7.6% at 0.02). Other effects at 0.01 after the 6 s sample: rainbow_cycle 15%, fire 14%, raindrops and level_up 19%, portal 11%, storm 13%, twinkle_multi 10%, radial_burst and beacon 8%, pong 7%, snowfall 5%, lighthouse 4%; christmas, candy_cane, plane_sweep, spiral_3d and lightning none.
  - In the firmware the accuracy is a persisted state setting (`colour_accuracy` in `POST /state`, HA's "Colour accuracy" number); `GET /` → `brightness_floor` and HA's "Brightness floor in effect" and "Colour floor" sensors say where the slider's bottom is and why. `render.py --colour-accuracy`, the viewer and the web export (`colour_floor_pct` per effect) apply it too.
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
| `./render.py <effect>` | Headless render: `renders/<effect>/{anim.gif,sheet.png,metrics.json}`. Options: `--seconds --fps --seed --param k=v --brightness --floor --colour-accuracy --limit-a --ma-per-channel --gamma --dither spatial|temporal|none --white R,G,B --view raw --mp4 --watch`. `metrics.json` reports the cap (`limit_a`, `budget_channel_sum`, `capped`, `cap_engaged_rate`) and estimated current in amps (`current_a` mean and peak, `uncapped_peak_a`), plus `per_frame.current_a` and `per_frame.capped`. |
| `./render.py all` | Every effect, plus `renders/overview.png` and `renders/summary.json`. |
| `micropython tools/mp_check.py` | Allocation check under MicroPython. |
| `micropython tools/output_check.py` | Output stage: native vs Python parity, allocation, timing. |
| `uv run --with pytest pytest tests/effects` | Output stage behaviour (the brightness and colour floors, no split below `DITHER_MIN`, cap, rounding, dithering). |
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
| `GET /effects` | `{"effects": [{"name", "duration_ms", "needs_coords", "params": {k: {"default", "min", "max", "value"}}, "overlay", "eligible", "error"}], "running", "fx_error"}` |
| `PUT` or `POST /effects/<name>` | Auth required (401). If the device has `UPDATE_KEY`, `X-Signature` must verify before any of the code runs (403). The body is module source, or `.mpy` (sniffed, or `?format=mpy`), at most 32 KB. Optional `X-Content-SHA256`; `?activate=1` also selects the effect. The device compile-checks the module, imports it under a staging name, builds `Effect(ctx)` and renders t=0 and t=33. Only then does it replace the overlay (`/fx/fx_<name>.py`) and hot-reload. Replies `200 {"effect", "overlay": true, "activated", "duration_ms", "params", "needs_coords"}`, or `400 {"error"}` and keeps the previous version. |
| `DELETE /effects/<name>` | Auth required. Removes the overlay. |

Limits:
- Only effect modules can be pushed. Shared modules (`_util`, `_output`, `_coords`) and `__init__` ship with firmware.
- A pushed effect with a new name is added without editing `EFFECTS`.
- Installing new firmware wipes all overlays.
