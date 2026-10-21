# Web animation export

`export_web.py` runs the registered Python effects through `render.run_effect` and exports the **raw effect frames**, equivalent to `render.py --view raw`. The web player needs only coordinates and RGB bytes: no Python runtime or JavaScript effect implementations. Only `effects.EFFECTS` is exported, in registry order; disabled effects are excluded.

```sh
./export_web.py --out renders/web --archive renders/tree-effects.tar.gz --sheet renders/web-sheet.png

# Decode an existing export, without invoking any effect:
./export_web.py --preview renders/web --sheet renders/web-sheet.png
```

The sheet reads the manifest and binaries from disk, verifies lengths and SHA-256, and draws front/side/top views plus an LED-index/time strip per effect. Unplaced LEDs remain in the binary but are omitted from the 3D views. The sheet's display gamma and glow are presentation only, not stored in the binaries.

## Format 1

The archive is gzip-compressed tar, with `manifest.json` and one `<name>.bin` at its root. There is no directory prefix or preview image inside the archive. Each binary is headerless uint8 RGB in **frame, LED, channel** order. No endianness conversion is needed. Byte offset `(frame * leds + led) * 3` gives that LED's R, G and B. Binary length must be exactly `effect.frames * manifest.leds * 3`.

The required playback fields are:

```json
{
  "format": 1,
  "source": {"commit": "<40-character checkout HEAD SHA>"},
  "fps": 20,
  "leds": 500,
  "output": {"applied": false, "brightness": 255, "limit_a": 1.5, "ma_per_channel": 8, "gamma": null},
  "coords": [[0.1, 0.5, -0.1], null],
  "seed": 0,
  "encoding": "rgb8",
  "frame_stage": "raw",
  "effects": [
    {
      "name": "rainbow_cycle",
      "label": "Rainbow cycle",
      "frames": 94,
      "file": "rainbow_cycle.bin",
      "byte_length": 141000,
      "gzip_bytes": 50837,
      "sha256": "<64-character hash of the uncompressed binary>",
      "params": {"speed": 1.0},
      "loop": {"mode": "period", "source_start_frame": 0, "crossfade_frames": 0}
    }
  ]
}
```

The illustrative `coords` above is shortened: the real array has exactly `leds` entries, copied from `coordinates/coords.txt`, rounded to four decimals. Missing positions are JSON `null`; indices are never compacted. Coordinates are in the effect contract frame: y up from 0 at the lowest LED to 1 at the highest, x/z in the same units, origin on the trunk, right-handed with +x to the front camera's image right and the front camera on +z. The exporter fails on a missing/invalid calibrated file; it never uses legacy coordinates.

All effects share `fps`. `frames / fps` is the loop duration (a one-frame effect is simply static). `file` is a bare name matching `^[a-z0-9_]+\.bin$`. Consumers should accept additive metadata fields; incompatible changes will increment `format`. `source.commit` identifies checkout HEAD, so development exports made with uncommitted edits can differ from that commit; release exports come from clean CI checkouts.

## Brightness and colour

The agreed web format uses **raw linear effect RGB**. It does not bake in the power cap, brightness, gamma, spatial dithering, white balance, or wire colour order. This avoids making a screen inherit the tree's electrical limits and spatial dithering. `frame_stage: "raw"` and `output.applied: false` make this explicit. The other `output` fields record the renderer's reference settings: brightness 255/255, the default 1.5 A limit at 8 mA/channel, and gamma disabled. They do **not** describe a transform applied to the binary. No device configuration or credentials are read.

For screen rendering, interpret `channel / 255` as linear intensity and convert to sRGB for display, or use the renderer's approximate display curve `pow(channel / 255, 1 / 2.2)`. Apply this once: a graphics pipeline that already converts linear output to sRGB does not need a second gamma conversion. A web brightness slider can scale linear RGB before that conversion. The bytes represent the same effect code and default parameters as the tree, but do not claim to simulate its measured physical brightness or current state. MicroPython and CPython also use different random generators.

## Loop policy and determinism

Every effect starts with its own `random.Random(0)`, default parameters, and calls at `t_ms = frame_index * 1000 // 20`, matching `render.py`. Files and archive contain no wall clock timestamps, absolute paths, owner names or device data. Two runs of the same source and Python/compression environment produce byte-identical manifests, binaries and archives; CI uses Python 3.12.

- `period`: use the module's `PERIOD_MS` at default speed 1, when representable at 20 fps. Render one extra endpoint, require it to equal frame zero, then omit that duplicate. `DURATION_MS` is the firmware's auto-cycle dwell time and is never treated as a period.
- `static`: warm white stores one frame, after checking two consecutive renders agree.
- `crossfade`: random/non-closing effects render ten seconds and produce an eight-second loop with a two-second overlap. For `N = 160`, `C = 40`, source frames `R`, output begins with `R[C:N]`. The final C frames blend `R[N+j]` with `R[j]` using weight `(j+1)/C`, integer RGB rounded down. The final output frame is exactly `R[C-1]`, and frame zero is `R[C]`, so wrapping is one ordinary source-frame step. The rest of the loop preserves unmodified effect frames. Blending is in linear RGB and is the only modification to the raw frames.

`loop.source_start_frame` identifies the unmodified first frame in a `render.py` run with the same fps/seed. `loop.crossfade_frames` says how many ending frames are blended. Natural periods preserve any intentional abrupt transitions in the effect itself. Tests verify the full unmodified portion, crossfade arithmetic, and closure against actual renders for every effect.

## Sizes

Measured with the calibrated 500-LED coordinates, 20 fps, seed 0. Gzip bytes are level 9 for each binary individually, without filename/timestamp headers; binaries inside the tar are uncompressed. The exporter fails if any effect exceeds 300,000 gzip bytes so growth is caught before publication.

| Effect | Frames | Seconds | Binary bytes | Gzip bytes |
|---|---:|---:|---:|---:|
| rainbow_cycle | 94 | 4.7 | 141,000 | 50,837 |
| christmas | 96 | 4.8 | 144,000 | 6,278 |
| candy_cane | 200 | 10 | 300,000 | 5,097 |
| twinkle_multi | 160 | 8 | 240,000 | 116,878 |
| warm_white | 1 | static | 1,500 | 34 |
| plane_sweep | 106 | 5.3 | 159,000 | 6,851 |
| radial_burst | 66 | 3.3 | 99,000 | 22,189 |
| spiral_3d | 80 | 4 | 120,000 | 14,076 |
| lighthouse | 100 | 5 | 150,000 | 23,067 |
| raindrops | 160 | 8 | 240,000 | 62,638 |
| snowfall | 160 | 8 | 240,000 | 21,150 |
| pong | 160 | 8 | 240,000 | 16,558 |
| level_up | 160 | 8 | 240,000 | 4,356 |
| beacon | 204 | 10.2 | 306,000 | 92,672 |
| tetris | 160 | 8 | 240,000 | 4,498 |

The complete archive is approximately **463 KB** (463,045 bytes with the 15 current effects, including the manifest; 342,686 bytes in the initial run with 11). Its exact size also depends on the recorded source commit. Preview images are CI artifacts/PR attachments, not release assets.

## Build-time download

`ldn-house/tree` is **public** (checked 19 October 2026). Once this change lands on main and CI publishes, the web build can fetch without credentials:

```sh
curl --fail --location --retry 3 \
  https://github.com/ldn-house/tree/releases/download/firmware-main/tree-effects.tar.gz \
  --output tree-effects.tar.gz
mkdir -p tree-effects
tar -xzf tree-effects.tar.gz -C tree-effects
```

Fetch at build time, validate `format`, coordinates, filenames, lengths and checksums, then serve the extracted manifest and binaries as static web assets. No browser-to-GitHub request, GitHub token, or effect runtime is needed. Use `source.commit` (or the archive hash) as a cache key: the rolling URL changes in place. A build may deliberately pin an archive for reproducibility.

After loading one binary, the player can select `floor(elapsedSeconds * manifest.fps) % effect.frames` and read each LED's RGB at the offset above. Interpolation to the following frame, including last-to-first, is optional. Skip `null` coordinates when drawing, while keeping their positions in the byte array.

The existing Firmware workflow builds/tests the export on PRs, uploads an inspectable CI artifact, and publishes the single archive with `--clobber` only after a trusted push to this repository's main branch. `publish-web` waits for the existing firmware publication but is not a dependency of it: exporter failures cannot block or modify `tree-firmware.bin` or the firmware `manifest.json`. Web uploads have their own concurrency group so they cannot cancel queued firmware jobs, with `queue: max` so an older run cannot replace the newest pending web upload. They check that main still matches their commit before uploading and only touch the web archive. Tagged firmware releases and PRs do not publish a web asset. A failed export leaves the previous web archive in place; check its embedded `source.commit` for freshness independently of the rolling release tag, which firmware publication can move.

## Validation

```sh
# Keep test files in the worktree's ignored scratch directory.
uv run --python 3.12 --with pytest --with numpy --with pillow \
  python -m pytest tests/test_web_export.py --basetemp=.scratch/web-export-tests -q
```

Tests export twice in separate processes and compare every file and the archive, check binary dimensions/hashes/size limits and registry membership, compare all effects with `render.run_effect(..., keep_raw=True)`, verify loop joins, and compare `render.py --view raw`'s contact sheet pixels with a sheet made from exported bytes. The preview test forbids effect loading/rendering to prove that it decodes the files. `render.py` currently records the wire-output hash in metrics even for `--view raw`, so raw parity uses its actual raw buffer and the CLI image, not that metrics hash.
