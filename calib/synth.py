"""Synthetic datasets: camera rigs, still captures (capture.py layout) and truth files."""

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera, phone_camera
from .layout import Layout
from .render import (SimParams, Sensor, View, ambient_series, encode_jpeg, make_view,
                     radiance)


def make_rig(kind: str, layout: Layout, n_views: int = 4, seed: int = 0,
             yaw_jitter: float = 12.0, scale: float = 1.0) -> list[Camera]:
    """Cameras roughly evenly spaced around the tree, handheld-style.

    webcam   608x1080 portrait crop of a 1080p webcam (fx about 1360 px) at
             about 3.3 m, like the real captures
    phone    1080x1920 portrait phone main camera (26 mm equivalent) at about 1.9 m
    """
    rng = np.random.default_rng(seed)
    top = layout.points[:, 1].max()
    cams = []
    for k in range(n_views):
        yaw = 360.0 * k / n_views + rng.uniform(-yaw_jitter, yaw_jitter)
        if kind == "webcam":
            w, h = int(608 * scale), int(1080 * scale)
            hfov = np.degrees(2 * np.arctan(304 / 1360))
            dist = rng.uniform(3.0, 3.6)
            k1 = 0.06
        elif kind == "phone":
            w, h = int(1080 * scale), int(1920 * scale)
            hfov = 45.6
            dist = rng.uniform(1.7, 2.3)
            k1 = 0.02
        else:
            raise ValueError(kind)
        cam = phone_camera(yaw, dist, rng.uniform(1.0, 1.5), top * rng.uniform(0.45, 0.55),
                           width=w, height_px=h, hfov_short_deg=hfov,
                           roll_deg=rng.uniform(-3, 3), k1=k1, name=f"view{k}")
        # Small principal-point offset, as real lenses have.
        cam.cx += rng.uniform(-0.01, 0.01) * w
        cam.cy += rng.uniform(-0.01, 0.01) * h
        cams.append(cam)
    return cams


def truth_dict(layout: Layout, views: list[View], extra: dict | None = None) -> dict:
    return {
        "layout": layout.name,
        "pitch": layout.pitch,
        "points": layout.points.tolist(),
        "views": [{
            "camera": v.cam.to_dict(),
            "uv": np.nan_to_num(v.uv, nan=-1).tolist(),
            "visible": v.visible.tolist(),
            "transmission": v.trans.tolist(),
            **getattr(v, "extra", {}),
        } for v in views],
        **(extra or {}),
    }


def lag_schedule(n_frames: int, lag: int, repeat_p: float, skip_p: float, rng) -> np.ndarray:
    """Displayed-state index captured by each frame, for a racing capture loop.

    Frame k should capture state k + 1 (state 0 is the pre-focus pattern). With
    probability repeat_p the camera returns the previous state again (lag grows
    by one); with skip_p it catches up by one (a state is never captured).
    """
    shown = np.zeros(n_frames, int)
    cur = lag
    for k in range(n_frames):
        if k > 1:
            r = rng.random()
            if r < repeat_p:
                cur += 1
            elif r < repeat_p + skip_p and cur > 0:
                cur -= 1
        shown[k] = k + 1 - cur
    return shown


def _render_still_view(job):
    layout, cam, params, out, mode, level, settle, lag, binary_level, vi = job[:10]
    repeat_p, skip_p = job[10] if len(job) > 10 else (0.0, 0.0)
    view = make_view(layout, cam, params, np.random.default_rng(params.seed * 100 + vi))
    sensor = Sensor(params, np.random.default_rng(params.seed * 100 + vi + 50))
    rng = np.random.default_rng(params.seed * 100 + vi + 70)
    n = layout.n
    name = cam.name or f"view{vi}"
    idx = np.arange(n)
    white = np.ones(3)
    # What the tree displays, in order: capture.py's pre-focus pattern (bit 3 at
    # 80), all off for the background, then each pattern.
    prefocus = ((idx >> 3) & 1)[:, None] * white * binary_level
    if mode == "sequential":
        patterns = [np.eye(n)[i][:, None] * white * level for i in range(n)]
        names = [f"led_{i:04d}" for i in range(n)]
        sub = "sequential"
    elif mode == "binary":
        patterns = [((idx >> b) & 1)[:, None] * white * binary_level for b in range(9)]
        names = [f"bit_{b}" for b in range(9)]
        sub = "binary"
    else:
        raise ValueError(mode)
    shown = [prefocus, np.zeros((n, 3))] + patterns
    captured = ["background"] + names
    d = out / name / sub
    d.mkdir(parents=True, exist_ok=True)
    amb = ambient_series(params, len(captured), rng)
    schedule = lag_schedule(len(captured), lag, repeat_p, skip_p, rng)
    for k, fname in enumerate(captured):
        img = sensor.expose(radiance(view, shown[schedule[k]], params, amb[k]), settle=settle)
        img = encode_jpeg(img, params.jpeg_quality)
        cv2.imwrite(str(d / f"{fname}.jpg"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    meta = {"angle": name, "mode": mode, "host": "synthetic", "led_count": n,
            "range": {"start": 0, "end": n}, "rotation": "none", "synthetic": True}
    (out / name / "metadata.json").write_text(json.dumps(meta, indent=2))
    # LED shown by each led_XXXX frame (-1 = none), for scoring index recovery.
    view.extra = {"frame_led": (schedule[1:] - 2).tolist()}
    return view


def render_stills(layout: Layout, cams: list[Camera], params: SimParams, out: Path,
                  mode: str = "sequential", level: int = 255, settle: int = 6,
                  lag: int = 0, binary_level: int = 80, workers: int = 4,
                  repeat_p: float = 0.0, skip_p: float = 0.0) -> list[View]:
    """Write captures/<view>/{sequential,binary}/*.jpg like capture.py, plus truth.json.

    lag simulates the real capture bug (0 or 1): frame i shows the pattern
    meant for frame i - lag, and the background shows the pre-focus pattern.
    """
    out = Path(out)
    jobs = [(layout, cam, params, out, mode, level, settle, lag, binary_level, vi, (repeat_p, skip_p))
            for vi, cam in enumerate(cams)]
    with ProcessPoolExecutor(workers) as ex:
        views = list(ex.map(_render_still_view, jobs))
    (out / "truth.json").write_text(json.dumps(truth_dict(layout, views, {
        "mode": mode, "lag": lag, "repeat_p": repeat_p, "skip_p": skip_p})))
    return views


def load_truth(path: Path) -> dict:
    t = json.loads(Path(path).read_text())
    t["points"] = np.array(t["points"])
    for v in t["views"]:
        v["camera"] = Camera.from_dict(v["camera"])
        v["uv"] = np.array(v["uv"], float)
        v["uv"][v["uv"][:, 0] < 0] = np.nan
        v["visible"] = np.array(v["visible"], bool)
    return t
