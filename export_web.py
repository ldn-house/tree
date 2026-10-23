#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Export deterministic RGB loops for web playback. See docs/web-export.md.

    ./export_web.py --out renders/web --archive renders/tree-effects.tar.gz
    ./export_web.py --preview renders/web --sheet renders/web-sheet.png
"""

import argparse
import gzip
import hashlib
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import effects
import render
from effects import _coords, _output

FPS = 20
SEED = 0
BRIGHTNESS = 1.0
LOOP_SECONDS = 8
FADE_SECONDS = 2
MAX_GZIP_BYTES = 300_000
COORDS_PATH = render.ROOT / "coordinates/coords.txt"


def calibrated_coords():
    """Require the calibrated file; never silently fall back or truncate it."""
    coords = _coords.parse(COORDS_PATH.read_text())
    if len(coords) != render.LED_COUNT or not any(c is not None for c in coords):
        raise ValueError("coordinates/coords.txt must contain 500 positions/nulls")
    if any(not all(math.isfinite(v) for v in c) or not 0 <= c[1] <= 1
           for c in coords if c is not None):
        raise ValueError("coordinates must be finite with y in [0, 1]")
    return coords


def render_frames(name, count, coords):
    result = render.run_effect(
        name, seconds=count / FPS, fps=FPS, seed=SEED, coords=coords,
        brightness=BRIGHTNESS, limit_a=_output.LIMIT_A,
        ma_per_channel=_output.MA_PER_CHANNEL, gamma=None, keep_raw=True,
    )
    metrics = result["metrics"]
    if metrics["exceptions"] or len(result["raw"]) != count:
        raise RuntimeError(f"{name}: incomplete render: {metrics['exceptions']}")
    return result["raw"]


def crossfade(frames, count, overlap):
    """Overlap the end with the moving head, then continue at head[overlap].

    Blend in linear effect RGB with integer weights, rounding down.
    No brightness, power cap, gamma, dithering or wire order is baked in.
    """
    loop = frames[overlap:count + overlap].copy()
    weight = np.arange(1, overlap + 1, dtype=np.int32)[:, None, None]
    tail = frames[count:count + overlap].astype(np.int32)
    head = frames[:overlap].astype(np.int32)
    loop[-overlap:] = (tail * (overlap - weight) + head * weight) // overlap
    return loop


def effect_loop(name, coords):
    module = effects.load(name)
    # DURATION_MS is an auto-cycle dwell time, NOT an animation period.
    if name == "warm_white":
        frames = render_frames(name, 2, coords)
        if np.array_equal(frames[0], frames[1]):
            return frames[:1], {"mode": "static", "source_start_frame": 0,
                                "crossfade_frames": 0}
    period = getattr(module, "PERIOD_MS", None)
    speed = effects.defaults(module).get("speed", 1.0)
    if period and speed == 1.0 and period * FPS % 1000 == 0:
        count = period * FPS // 1000
        frames = render_frames(name, count + 1, coords)
        # Check closure against the real code, before any output transform.
        if np.array_equal(frames[0], frames[-1]):
            return frames[:-1], {"mode": "period", "source_start_frame": 0,
                                 "crossfade_frames": 0}
    count, overlap = LOOP_SECONDS * FPS, FADE_SECONDS * FPS
    frames = render_frames(name, count + overlap, coords)
    return crossfade(frames, count, overlap), {
        "mode": "crossfade", "source_start_frame": overlap,
        "crossfade_frames": overlap,
    }


def gzip_bytes(data):
    # GzipFile fixes both the timestamp and filename (unlike gzip's defaults).
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb", filename="", mtime=0, compresslevel=9) as gz:
        gz.write(data)
    return out.getvalue()


def write_archive(out, manifest, path):
    """Only manifest-listed files; no stale exports, previews or local data."""
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for name in ["manifest.json"] + [e["file"] for e in manifest["effects"]]:
            contents = (out / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(contents)
            info.mode = 0o644
            # TarInfo defaults: epoch timestamp, uid/gid 0, empty owner names.
            archive.addfile(info, io.BytesIO(contents))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip_bytes(data.getvalue()))


def export(out, archive=None):
    out = Path(out)
    coords = calibrated_coords()
    manifest = {
        "format": 1,
        "source": {"commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=render.ROOT, text=True).strip()},
        "leds": render.LED_COUNT,
        "fps": FPS,
        "seed": SEED,
        "encoding": "rgb8",
        "frame_stage": "raw",
        "output": {"applied": False, "brightness": round(BRIGHTNESS * 255), "limit_a": _output.LIMIT_A,
                   "ma_per_channel": _output.MA_PER_CHANNEL, "gamma": None,
                   # A player that dims should map brightness 1..255 onto
                   # floor..100% as the tree does (effects/_output.py).
                   "brightness_floor_pct": _output.FLOOR_PCT},
        "coords": [None if c is None else [round(v, 4) for v in c] for c in coords],
        "effects": [],
    }
    out.mkdir(parents=True, exist_ok=True)
    # Do not leave a valid-looking manifest if this export fails part way.
    (out / "manifest.json").unlink(missing_ok=True)
    for name in effects.EFFECTS:
        frames, loop = effect_loop(name, coords)
        data = frames.tobytes(order="C")
        compressed_size = len(gzip_bytes(data))
        if compressed_size > MAX_GZIP_BYTES:
            raise ValueError(f"{name}: {compressed_size} gzip bytes exceeds {MAX_GZIP_BYTES}")
        filename = f"{name}.bin"
        (out / filename).write_bytes(data)
        entry = {
            "name": name,
            "label": name.replace("_", " ").capitalize().replace("3d", "3D"),
            "frames": len(frames),
            "file": filename,
            "byte_length": len(data),
            "gzip_bytes": compressed_size,
            "sha256": hashlib.sha256(data).hexdigest(),
            "params": effects.defaults(effects.load(name)),
            "loop": loop,
        }
        manifest["effects"].append(entry)
        print(f"{name:16s} {len(frames):3d} frames  {len(data):7,d} bytes  "
              f"{compressed_size:7,d} gzip bytes  {loop['mode']}")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    if archive is not None:
        write_archive(out, manifest, Path(archive))
        print(f"archive: {archive} ({Path(archive).stat().st_size:,} bytes)")
    return manifest


def preview(out, path):
    """Decode only the exported manifest/binaries; never invoke effects."""
    out, path = Path(out), Path(path)
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["format"] != 1 or manifest["encoding"] != "rgb8":
        raise ValueError("unsupported web export format")
    coords = manifest["coords"]
    # Omit unplaced LEDs rather than inventing positions for the contact sheet.
    known = np.array([c is not None for c in coords])
    views = render.Views(np.array([c for c in coords if c is not None]))
    rows = []
    for effect in manifest["effects"]:
        data = (out / effect["file"]).read_bytes()
        if len(data) != effect["frames"] * manifest["leds"] * 3:
            raise ValueError(f"{effect['name']}: binary length does not match manifest")
        if hashlib.sha256(data).hexdigest() != effect["sha256"]:
            raise ValueError(f"{effect['name']}: checksum does not match manifest")
        frames = np.frombuffer(data, dtype=np.uint8).reshape(-1, manifest["leds"], 3)
        indices = np.linspace(0, len(frames) - 1, 4).round().astype(int)
        row = Image.new("RGB", (960, 230), render.BG)
        draw = ImageDraw.Draw(row)
        draw.text((8, 4), f"{effect['label']}  |  {len(frames)} frames @ {manifest['fps']} fps"
                  f"  |  {effect['loop']['mode']}  |  decoded RGB", fill=(230, 230, 230))
        for col, index in enumerate(indices):
            tile = views.frame(frames[index][known], f"frame {index}")
            row.paste(tile.resize((240, 127), Image.Resampling.LANCZOS), (col * 240, 24))
        timeline = render.timeline_image(frames, width=944, height=68)
        row.paste(timeline, (8, 156))
        rows.append(row)
    sheet = Image.new("RGB", (960, 32 + len(rows) * 230), render.BG)
    ImageDraw.Draw(sheet).text((8, 8), "Exported binary playback | front / side / top | display gamma 2.2",
                              fill=(230, 230, 230))
    for i, row in enumerate(rows):
        sheet.paste(row, (0, 32 + i * 230))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("renders/web"))
    parser.add_argument("--archive", type=Path, help="also write a deterministic .tar.gz")
    parser.add_argument("--preview", type=Path, help="decode an existing export instead of rendering effects")
    parser.add_argument("--sheet", type=Path, help="write a contact sheet decoded from the exported files")
    args = parser.parse_args()
    if args.preview and (not args.sheet or args.archive):
        parser.error("--preview requires --sheet and cannot be used with --archive")
    if not args.preview:
        export(args.out, args.archive)
    if args.sheet:
        preview(args.preview or args.out, args.sheet)


if __name__ == "__main__":
    main()
