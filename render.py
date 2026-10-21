#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Headless renderer for tree effects.

Runs an effect from effects/ under CPython through the same output stage as
the firmware (power cap -> brightness above the floor -> white balance ->
gamma -> dithering) and writes, per effect:

    renders/<effect>/anim.gif    front | side | top views of the real coordinates
    renders/<effect>/sheet.png   contact sheet of evenly spaced frames, plus an
                                 LED-index x time strip of the whole run
    renders/<effect>/metrics.json

Output is deterministic for a given effect source, seed and options, so the
frames_sha256 in metrics.json can be compared between runs.

Usage:
    ./render.py rainbow_cycle
    ./render.py spiral_3d --seconds 8 --fps 30 --param twists=2 --brightness 0.5
    ./render.py all                 # every registered effect + renders/overview.png
    ./render.py spiral_3d --watch   # re-render whenever effects/ changes
"""

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import effects  # noqa: E402
from effects import _coords, _output  # noqa: E402

LED_COUNT = 500
DEFAULT_COORDS = ("coordinates/coords.txt", "coordinates/coords_compact.txt")

PANEL_W, PANEL_H, HEADER_H = 240, 360, 22
BG = (6, 8, 12)


# ─── Running effects ──────────────────────────────────────────────────────────


def find_coords(path=None, n=LED_COUNT):
    """Return (coords or None, source path or None)."""
    candidates = [path] if path else [str(ROOT / p) for p in DEFAULT_COORDS]
    for p in candidates:
        if p and os.path.exists(p):
            coords = _coords.load(p, n)
            if coords is not None:
                return coords, os.path.relpath(p, ROOT)
    return None, None


def parse_value(text):
    for kind in (int, float):
        try:
            return kind(text)
        except ValueError:
            pass
    return text


def build_params(module, overrides):
    """Merge --param overrides into the effect's defaults; returns
    (params, warnings)."""
    params = effects.defaults(module)
    spec = getattr(module, "PARAMS", {})
    warnings = []
    for key, value in overrides.items():
        if key in spec:
            lo, hi = spec[key][1], spec[key][2]
            if isinstance(value, (int, float)) and not lo <= value <= hi:
                warnings.append(f"param {key}={value} outside [{lo}, {hi}]")
        else:
            warnings.append(f"unknown param {key!r} (known: {', '.join(spec) or 'none'})")
        params[key] = value
    return params, warnings


def run_effect(name, seconds=6.0, fps=20, seed=0, overrides=None, brightness=1.0,
               limit_a=_output.LIMIT_A, ma_per_channel=_output.MA_PER_CHANNEL, gamma=None,
               coords=None, n=LED_COUNT, keep_raw=False, dither="spatial", white=None,
               floor=_output.FLOOR_PCT):
    """Render an effect headlessly. Returns a dict with output frames
    (uint8 array frames x n x 3, after the output stage) and metrics."""
    module = effects.load(name)
    params, warnings = build_params(module, overrides or {})
    if getattr(module, "NEEDS_COORDS", False) and not any(coords or []):
        warnings.append("effect needs coordinates but none are loaded")
    ctx = effects.Ctx(n, coords, params, random.Random(seed))
    if float(ma_per_channel).is_integer():
        ma_per_channel = int(ma_per_channel)
    budget = int(_output.budget_for_current(limit_a, ma_per_channel))
    stage = _output.Output(n, budget=budget, gamma=gamma, dither=dither, white=white, floor=floor)
    bri = max(0, min(255, round(brightness * 255)))
    count = max(1, round(seconds * fps))
    raw = bytearray(3 * n)
    out = bytearray(3 * n)
    outs = np.zeros((count, n, 3), np.uint8)
    raws = np.zeros((count, n, 3), np.uint8) if keep_raw else None
    capped = np.zeros(count, bool)
    factor = np.full(count, 4096, np.int64)
    raw_totals = np.zeros(count, np.int64)
    exceptions = []
    render_s = []
    done = 0
    try:
        effect = module.Effect(ctx)
    except Exception:
        exceptions.append({"frame": None, "t_ms": None, "where": "__init__",
                           "error": traceback.format_exc(limit=8)})
        effect = None
    for f in range(count if effect else 0):
        t_ms = f * 1000 // fps
        t0 = time.perf_counter()
        try:
            result = effect.render(t_ms, raw)
            if result is not None:
                raise TypeError(f"render() returned {type(result).__name__}, expected None")
            if not isinstance(raw, bytearray) or len(raw) != 3 * n:
                raise TypeError("render() resized or replaced the frame buffer")
        except Exception:
            exceptions.append({"frame": f, "t_ms": t_ms, "where": "render",
                               "error": traceback.format_exc(limit=8)})
            break
        render_s.append(time.perf_counter() - t0)
        stage.process(raw, out, bri)
        outs[f] = np.frombuffer(out, np.uint8).reshape(n, 3)
        if keep_raw:
            raws[f] = np.frombuffer(raw, np.uint8).reshape(n, 3)
        capped[f] = stage.capped
        factor[f] = stage.factor
        raw_totals[f] = stage.raw_total
        done += 1
    outs = outs[:done]
    if keep_raw:
        raws = raws[:done]

    out_totals = outs.reshape(done, -1).sum(axis=1).astype(np.int64) if done else np.zeros(0)
    lit = (outs.max(axis=2) > 0).mean(axis=1) if done else np.zeros(0)
    # Estimated current per frame from the bytes actually output (after cap,
    # brightness, white balance, gamma and dithering), as the firmware does.
    current = np.array([_output.current_a(int(out_totals[f]), ma_per_channel) for f in range(done)])
    # Dithering may put a frame up to 1 LSB per lit channel above its exact
    # target; on average the output stays within the budget.
    lit_channels = (outs > 0).reshape(done, -1).sum(axis=1) if done else np.zeros(0)
    uncapped = np.array([_output.current_a(int(raw_totals[f]) * stage.level(bri) // 65280, ma_per_channel)
                         for f in range(done)])

    def stat(a, digits=3):
        a = np.asarray(a, float)
        if not len(a):
            return None
        return {"mean": round(float(a.mean()), digits), "min": round(float(a.min()), digits),
                "max": round(float(a.max()), digits)}

    metrics = {
        "effect": name,
        "frames": int(done),
        "frames_requested": int(count),
        "fps": fps,
        "seconds": seconds,
        "seed": seed,
        "params": params,
        "brightness": round(bri / 255, 4),
        "brightness_floor_pct": floor,
        "brightness_level": round(stage.level(bri) / 65280, 4),
        "gamma": gamma,
        "dither": dither or None,
        "white": list(white) if white else None,
        "led_count": n,
        "coords_loaded": sum(1 for c in (coords or []) if c is not None),
        "power": {
            "limit_a": limit_a,
            "ma_per_channel": ma_per_channel,
            "budget_channel_sum": budget,
            "capped": bool(capped[:done].any()),
            "cap_engaged_rate": round(float(capped[:done].mean()), 4) if done else None,
            "cap_factor": stat(factor[:done] / 4096),
            "current_a": {"mean": round(float(current.mean()), 3) if done else None,
                          "peak": round(float(current.max()), 3) if done else None},
            "uncapped_peak_a": round(float(uncapped.max()), 3) if done else None,
            "over_budget_frames": int((out_totals > budget + lit_channels).sum()),
            "mean_over_budget": round(float(out_totals.mean() - budget), 2) if done else None,
            "max_over_budget": int((out_totals - budget).max()) if done else None,
            "per_frame": {"current_a": [round(float(c), 3) for c in current],
                          "capped": [bool(c) for c in capped[:done]]},
        },
        "lit_fraction": stat(lit),
        "all_black": bool(done and int(out_totals.max()) == 0),
        "black_frame_rate": round(float((out_totals == 0).mean()), 4) if done else None,
        "frames_sha256": hashlib.sha256(outs.tobytes()).hexdigest(),
        "exceptions": exceptions,
        "warnings": warnings,
    }
    perf = {"render_ms_mean": round(1000 * float(np.mean(render_s)), 2) if render_s else None,
            "render_ms_max": round(1000 * float(np.max(render_s)), 2) if render_s else None}
    return {"frames": outs, "raw": raws, "capped": capped[:done], "metrics": metrics,
            "perf": perf, "fps": fps}


# ─── Drawing ──────────────────────────────────────────────────────────────────


def display_positions(coords, n=LED_COUNT):
    """(n, 3) positions for drawing; LEDs without coordinates get a
    stand-in spiral so string-order effects still show up."""
    pos = np.zeros((n, 3))
    known = np.zeros(n, bool)
    for i in range(n):
        c = coords[i] if coords and i < len(coords) else None
        if c is not None:
            pos[i] = c
            known[i] = True
        else:
            h = i / n
            a = i * 2.39996
            r = 0.3 * (1 - h)
            pos[i] = (r * np.cos(a), h, r * np.sin(a))
    return pos, known


class Views:
    """Projects LEDs into front, side and top panels and splats a glow per
    LED. Front: camera on +z looking at the trunk (image right = +x). Side:
    camera on +x (image right = -z). Top: looking down (+z toward the bottom
    of the panel). Nearer LEDs draw slightly brighter."""

    def __init__(self, positions, display_gamma=2.2):
        self.display_gamma = display_gamma
        x, y, z = positions[:, 0], positions[:, 1], positions[:, 2]
        half = max(0.05, np.abs(positions[:, [0, 2]]).max()) * 1.08
        self.views = []
        for name, u, v, depth, vrange in (
            ("front", x, y, z, (-0.04, 1.04)),
            ("side", -z, y, x, (-0.04, 1.04)),
            ("top", x, z, y, (-half, half)),
        ):
            uspan = half if name != "top" else half
            if name == "top":
                scale = min(PANEL_W, PANEL_H) / (2 * half) * 0.92
                cx, cy = PANEL_W / 2, PANEL_H / 2
                px = cx + u * scale
                py = cy + v * scale
            else:
                scale = (PANEL_H - 16) / (vrange[1] - vrange[0])
                scale = min(scale, (PANEL_W - 12) / (2 * uspan))
                px = PANEL_W / 2 + u * scale
                py = PANEL_H - 8 - (v - vrange[0]) * scale
            dn = (depth - depth.min()) / (np.ptp(depth) or 1)
            self.views.append((name, px, py, 0.65 + 0.35 * dn, np.argsort(depth)))
        r = 7
        yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
        d2 = xx * xx + yy * yy
        self.kernel = (np.exp(-d2 / 7.0) * 0.9 + np.exp(-d2 / 1.2) * 0.8).astype(np.float32)
        self.kr = r

    def _panel(self, colors, view):
        name, px, py, near, order = view
        r = self.kr
        acc = np.zeros((PANEL_H + 2 * r, PANEL_W + 2 * r, 3), np.float32)
        lin = (colors / 255.0) ** (1 / self.display_gamma)
        ix = np.round(px).astype(int) + r
        iy = np.round(py).astype(int) + r
        ok = (ix >= r) & (ix < PANEL_W + r) & (iy >= r) & (iy < PANEL_H + r)
        base = np.zeros((PANEL_H, PANEL_W, 3), np.float32)
        for i in order:
            if not ok[i]:
                continue
            c = lin[i] * near[i]
            if c.max() <= 0:
                # Unlit LED: faint marker so the tree's shape stays visible.
                base[iy[i] - r, ix[i] - r] = np.maximum(base[iy[i] - r, ix[i] - r], 0.13)
                continue
            acc[iy[i] - r:iy[i] + r + 1, ix[i] - r:ix[i] + r + 1] += self.kernel[..., None] * c
        img = np.maximum(acc[r:-r, r:-r], base)
        img = 1 - np.exp(-img * 1.6)  # soft tone-map so overlapping glows saturate gracefully
        bg = np.array(BG, np.float32) / 255
        img = bg + (1 - bg) * img
        return (np.clip(img, 0, 1) * 255).astype(np.uint8)

    def frame(self, colors, header=""):
        panels = [self._panel(colors.astype(np.float32), v) for v in self.views]
        body = np.concatenate(
            [np.pad(p, ((0, 0), (0, 2 if k < 2 else 0), (0, 0)), constant_values=30)
             for k, p in enumerate(panels)], axis=1)
        img = Image.new("RGB", (body.shape[1], body.shape[0] + HEADER_H), BG)
        img.paste(Image.fromarray(body), (0, HEADER_H))
        d = ImageDraw.Draw(img)
        font = _font(12)
        for k, (name, *_rest) in enumerate(self.views):
            d.text((k * (PANEL_W + 2) + 6, HEADER_H + 4), name, fill=(120, 130, 150), font=font)
        d.text((6, 4), header, fill=(220, 220, 220), font=font)
        return img


_fonts = {}


def _font(size):
    if size not in _fonts:
        try:
            _fonts[size] = ImageFont.load_default(size=size)
        except TypeError:
            _fonts[size] = ImageFont.load_default()
    return _fonts[size]


def frame_header(result, f):
    m = result["metrics"]
    t = f / result["fps"]
    amps = m["power"]["per_frame"]["current_a"][f] if f < len(m["power"]["per_frame"]["current_a"]) else 0
    cap = "CAP" if result["capped"][f] else "   "
    return (f"{m['effect']}  t={t:5.2f}s  {cap}  ~{amps:.2f} A"
            f"  bri={m['brightness']:.2f}")


def timeline_image(frames, width=None, height=None):
    """LED index (x) x time (y, top = start) strip of the whole run."""
    disp = ((frames / 255.0) ** (1 / 2.2) * 255).astype(np.uint8)
    img = Image.fromarray(disp)
    if width or height:
        img = img.resize((width or img.width, height or img.height), Image.NEAREST)
    return img


def contact_sheet(result, views, cells=12, cols=3, scale=0.6):
    frames = result["frames"]
    count = len(frames)
    if count == 0:
        img = Image.new("RGB", (600, 120), BG)
        ImageDraw.Draw(img).text((10, 10), "no frames rendered:\n" + json.dumps(
            result["metrics"]["exceptions"])[:400], fill=(255, 80, 80), font=_font(12))
        return img
    idx = sorted(set(np.linspace(0, count - 1, min(cells, count)).round().astype(int)))
    thumbs = [views.frame(frames[i], frame_header(result, i)) for i in idx]
    tw, th = int(thumbs[0].width * scale), int(thumbs[0].height * scale)
    thumbs = [t.resize((tw, th), Image.LANCZOS) for t in thumbs]
    rows = (len(thumbs) + cols - 1) // cols
    m = result["metrics"]
    title_h = 40
    tl_h = 160
    sheet_w = cols * tw + (cols + 1) * 4
    sheet = Image.new("RGB", (sheet_w, title_h + rows * (th + 4) + tl_h + 30), BG)
    d = ImageDraw.Draw(sheet)
    p = m["power"]
    lf = m["lit_fraction"] or {}
    d.text((6, 4), f"{m['effect']}  {m['frames']} frames @ {m['fps']} fps  seed={m['seed']}  "
                   f"params={json.dumps(m['params'])}", fill=(230, 230, 230), font=_font(13))
    d.text((6, 22), f"cap engaged {100 * (p['cap_engaged_rate'] or 0):.0f}% of frames  "
                    f"current mean {p['current_a']['mean']:.2f} A / peak {p['current_a']['peak']:.2f} A "
                    f"(limit {p['limit_a']} A at {p['ma_per_channel']} mA/channel)  lit {lf.get('mean', 0):.0%}  "
                    f"exceptions {len(m['exceptions'])}", fill=(170, 180, 200), font=_font(12))
    for k, t in enumerate(thumbs):
        r, c = divmod(k, cols)
        sheet.paste(t, (4 + c * (tw + 4), title_h + r * (th + 4)))
    y = title_h + rows * (th + 4) + 4
    d.text((6, y), "timeline: LED index 0-499 left to right, time top to bottom",
           fill=(150, 160, 180), font=_font(12))
    sheet.paste(timeline_image(frames, sheet_w - 8, tl_h), (4, y + 20))
    return sheet


def write_gif(result, views, path, scale=0.6):
    frames = result["frames"]
    if not len(frames):
        return
    imgs = [views.frame(frames[i], frame_header(result, i)) for i in range(len(frames))]
    if scale != 1:
        size = (round(imgs[0].width * scale), round(imgs[0].height * scale))
        imgs = [im.resize(size, Image.LANCZOS) for im in imgs]
    # One shared palette keeps the GIF stable and small.
    pal = imgs[len(imgs) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    q = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in imgs]
    q[0].save(path, save_all=True, append_images=q[1:], duration=round(1000 / result["fps"]),
              loop=0, optimize=True, disposal=1)


def write_mp4(result, views, path):
    frames = result["frames"]
    ffmpeg = subprocess.run(["which", "ffmpeg"], capture_output=True, text=True).stdout.strip()
    if not ffmpeg:
        print("  (ffmpeg not found; skipping mp4)")
        return False
    first = views.frame(frames[0], "")
    w, h = first.width - first.width % 2, first.height - first.height % 2
    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{w}x{h}", "-r", str(result["fps"]), "-i", "-", "-pix_fmt", "yuv420p",
         "-c:v", "libx264", "-crf", "20", str(path)], stdin=subprocess.PIPE)
    for i in range(len(frames)):
        proc.stdin.write(np.asarray(views.frame(frames[i], frame_header(result, i)))[:h, :w].tobytes())
    proc.stdin.close()
    return proc.wait() == 0


# ─── CLI ──────────────────────────────────────────────────────────────────────


def render_one(name, args, coords, coords_src, overrides, out_root):
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    result = run_effect(name, args.seconds, args.fps, args.seed, overrides, args.brightness,
                        args.limit_a, args.ma_per_channel, args.gamma, coords, keep_raw=args.view == "raw",
                        dither=None if args.dither == "none" else args.dither, white=args.white,
                        floor=args.floor)
    if args.view == "raw" and result["raw"] is not None:
        result["frames"] = result["raw"]
    result["metrics"]["coords_source"] = coords_src
    result["metrics"]["view"] = args.view
    views = Views(display_positions(coords)[0])
    (out_dir / "metrics.json").write_text(json.dumps(result["metrics"], indent=2) + "\n")
    contact_sheet(result, views, cells=args.sheet_frames).save(out_dir / "sheet.png")
    if not args.no_anim:
        write_gif(result, views, out_dir / "anim.gif", args.gif_scale)
        if args.mp4:
            write_mp4(result, views, out_dir / "anim.mp4")
    return result


def overview(results, coords, path, per_row=5):
    """One row per effect: a few evenly spaced frames of the front view."""
    views = Views(display_positions(coords)[0])
    rows = []
    for res in results:
        frames = res["frames"]
        name = res["metrics"]["effect"]
        idx = np.linspace(0, max(0, len(frames) - 1), per_row).round().astype(int) if len(frames) else []
        cells = []
        for i in idx:
            panel = views._panel(frames[i].astype(np.float32), views.views[0])
            cells.append(Image.fromarray(panel).resize((PANEL_W // 2, PANEL_H // 2), Image.LANCZOS))
        row = Image.new("RGB", (130 + per_row * (PANEL_W // 2 + 3), PANEL_H // 2 + 4), BG)
        d = ImageDraw.Draw(row)
        m = res["metrics"]
        d.text((6, 6), name, fill=(230, 230, 230), font=_font(13))
        cap = m["power"]["cap_engaged_rate"]
        d.text((6, 26), f"cap {100 * (cap or 0):.0f}%\npeak {m['power']['current_a']['peak']:.2f} A\n"
                        f"lit {(m['lit_fraction'] or {}).get('mean', 0):.0%}"
                        + ("\nEXCEPTION" if m["exceptions"] else ""),
               fill=(160, 170, 190) if not m["exceptions"] else (255, 90, 90), font=_font(11))
        for k, c in enumerate(cells):
            row.paste(c, (130 + k * (PANEL_W // 2 + 3), 2))
        rows.append(row)
    sheet = Image.new("RGB", (max(r.width for r in rows), sum(r.height for r in rows)), BG)
    y = 0
    for r in rows:
        sheet.paste(r, (0, y))
        y += r.height
    sheet.save(path)


def summary_line(m):
    p = m["power"]
    status = "FAIL" if m["exceptions"] else "ok"
    return (f"{m['effect']:<15} {status:<4} frames={m['frames']:<4} cap={100 * (p['cap_engaged_rate'] or 0):3.0f}% "
            f"mean={p['current_a']['mean']:.2f}A peak={p['current_a']['peak']:.2f}A "
            f"lit={(m['lit_fraction'] or {}).get('mean', 0):.0%}")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("effect", help="effect name, or 'all' (choices: " + ", ".join(effects.EFFECTS) + ")")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--param", action="append", default=[], metavar="K=V",
                    help="override an effect param (repeatable)")
    ap.add_argument("--brightness", type=float, default=1.0,
                    help="0-1 (default 1); non-zero maps onto floor..1 as on the tree")
    ap.add_argument("--floor", type=float, default=_output.FLOOR_PCT,
                    help=f"brightness floor, percent (default {_output.FLOOR_PCT}; 0 for none)")
    ap.add_argument("--limit-a", type=float, default=_output.LIMIT_A,
                    help=f"power cap in amps at 12 V (default {_output.LIMIT_A})")
    ap.add_argument("--ma-per-channel", type=float, default=_output.MA_PER_CHANNEL,
                    help=f"current of one channel at 255, mA (default {_output.MA_PER_CHANNEL})")
    ap.add_argument("--gamma", type=float, default=None, help="output gamma (default: linear)")
    ap.add_argument("--dither", choices=["spatial", "temporal", "none"], default="spatial",
                    help="output dithering (default spatial: fixed per LED, no flicker)")
    ap.add_argument("--white", type=lambda v: tuple(float(x) for x in v.split(",")), default=None,
                    metavar="R,G,B", help="per-channel white balance, e.g. 1,0.85,0.7")
    ap.add_argument("--coords", help="coordinates file (default: coordinates/coords.txt, "
                                     "else the legacy coordinates/coords_compact.txt)")
    ap.add_argument("--no-coords", action="store_true", help="render as if uncalibrated")
    ap.add_argument("--out", default="renders", help="output directory (default renders/)")
    ap.add_argument("--view", choices=["output", "raw"], default="output",
                    help="draw the output-stage frame (default) or the raw effect frame")
    ap.add_argument("--sheet-frames", type=int, default=12)
    ap.add_argument("--no-anim", action="store_true", help="skip the GIF (faster)")
    ap.add_argument("--gif-scale", type=float, default=0.6, help="GIF size relative to the stills (default 0.6)")
    ap.add_argument("--mp4", action="store_true", help="also write anim.mp4 (needs ffmpeg)")
    ap.add_argument("--watch", action="store_true", help="re-render when effects/ changes")
    return ap.parse_args(argv)


def watch(argv):
    """Re-run this script (minus --watch) whenever a file in effects/ changes."""
    cmd = [sys.executable, str(Path(__file__).resolve())] + [a for a in argv if a != "--watch"]

    def snapshot():
        return {p: p.stat().st_mtime for p in (ROOT / "effects").glob("*.py")}

    seen = None
    print("watching effects/ (Ctrl-C to stop)", flush=True)
    try:
        while True:
            now = snapshot()
            if now != seen:
                if seen is not None:
                    changed = sorted(p.name for p in now if now.get(p) != seen.get(p))
                    print(f"\nchanged: {', '.join(changed)}", flush=True)
                seen = now
                subprocess.run(cmd)
            time.sleep(0.4)
    except KeyboardInterrupt:
        pass


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    args = parse_args(argv)
    if args.watch:
        return watch(argv)
    names = effects.EFFECTS if args.effect == "all" else [args.effect]
    for n in names:
        if n not in effects.EFFECTS:
            sys.exit(f"unknown effect {n!r}; registered: {', '.join(effects.EFFECTS)}")
    overrides = {}
    for kv in args.param:
        if "=" not in kv:
            sys.exit(f"--param expects K=V, got {kv!r}")
        k, v = kv.split("=", 1)
        overrides[k] = parse_value(v)
    coords, src = (None, None) if args.no_coords else find_coords(args.coords)
    out_root = Path(args.out)
    print(f"coords: {src or 'none'}; brightness {args.brightness}; cap {args.limit_a:g} A at {args.ma_per_channel:g} mA/channel")
    results = []
    failed = False
    for name in names:
        t0 = time.perf_counter()
        res = render_one(name, args, coords, src, overrides if len(names) == 1 else {}, out_root)
        results.append(res)
        m = res["metrics"]
        failed |= bool(m["exceptions"])
        print(summary_line(m) + f"  ({time.perf_counter() - t0:.1f}s, render {res['perf']['render_ms_mean']} ms/frame)")
        for w in m["warnings"]:
            print(f"  warning: {w}")
        for e in m["exceptions"]:
            print(f"  exception at frame {e['frame']}:\n{e['error']}")
        print(f"  -> {out_root / name}/{{sheet.png,anim.gif,metrics.json}}")
    if len(names) > 1:
        overview(results, coords, out_root / "overview.png")
        (out_root / "summary.json").write_text(json.dumps([r["metrics"] for r in results], indent=2) + "\n")
        print(f"overview: {out_root / 'overview.png'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
