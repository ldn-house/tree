"""Figures for docs/calibration-findings.md.

Real captures show the inside of a home, so real-data figures only ever show
residual images (frame minus the per-pixel reference: LED light only) or plots.
"""

import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .detect import StillDetector, fit_gain, linear_gray  # noqa: E402


def _to8(lin: np.ndarray, gain: float) -> np.ndarray:
    v = np.clip(lin * gain, 0, 1) ** (1 / 2.2)
    return (v * 255).astype(np.uint8)


def residual_crops(seq_dir: Path, leds: list[int], lag: int, half: int = 60, title: str = "") -> np.ndarray:
    """LED-only residual crops (frame minus fitted reference) around each LED."""
    frames = sorted(seq_dir.glob("led_*.jpg"))
    det = StillDetector("ref")
    det.prepare(frames)
    tiles = []
    for led in leds:
        img = cv2.imread(str(frames[led + lag]))
        lin = linear_gray(img)
        a, b = fit_gain(lin, det.ref)
        res = np.clip(lin - (a * det.ref + b), 0, None)
        y, x = np.unravel_index(np.argmax(cv2.GaussianBlur(res, (0, 0), 2)), res.shape)
        pad = np.pad(res, half)
        crop = pad[y:y + 2 * half, x:x + 2 * half]
        t = cv2.cvtColor(_to8(crop, 1.0 / max(crop.max(), 1e-6) * 1.5), cv2.COLOR_GRAY2BGR)
        t = cv2.resize(t, (240, 240), interpolation=cv2.INTER_NEAREST)
        cv2.putText(t, f"{title} LED {led}", (6, 18), 0, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    return np.hstack(tiles)


def sim_vs_real(real_dir: Path, synth_dir: Path, out: Path):
    real = residual_crops(real_dir, [60, 150, 240, 330, 420], lag=1, title="real")
    synth = residual_crops(synth_dir, [60, 150, 240, 330, 420], lag=0, title="synthetic")
    cv2.imwrite(str(out), np.vstack([real, synth]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def detection_overlay(synth_view_dir: Path, truth_uv: np.ndarray, dets: dict[str, np.ndarray],
                      out: Path, leds: list[int]):
    """Crops of synthetic frames with truth (green) and each detector's answer."""
    colours = {"local_contrast": (0, 0, 255), "diff": (255, 255, 0)}
    tiles = []
    for led in leds:
        img = cv2.imread(str(synth_view_dir / f"led_{led:04d}.jpg"))
        vis = cv2.convertScaleAbs(img, alpha=2.5)
        x, y = truth_uv[led]
        if not np.isfinite(x):
            continue
        for name, uv in dets.items():
            u = uv[led]
            if np.isfinite(u).all():
                cv2.circle(vis, (int(u[0]), int(u[1])), 7, colours.get(name, (255, 0, 255)), 2, cv2.LINE_AA)
        cv2.drawMarker(vis, (int(x), int(y)), (0, 255, 0), cv2.MARKER_CROSS, 14, 1, cv2.LINE_AA)
        h, w = vis.shape[:2]
        x0, y0 = int(np.clip(x - 90, 0, w - 180)), int(np.clip(y - 90, 0, h - 180))
        crop = cv2.resize(vis[y0:y0 + 180, x0:x0 + 180], (240, 240))
        cv2.putText(crop, f"LED {led}", (6, 18), 0, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(crop)
    legend = np.zeros((30, 240 * len(tiles), 3), np.uint8)
    cv2.putText(legend, "green +: truth   red: local_contrast (current)   cyan: diff (new)", (6, 20), 0, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(out), np.vstack([np.hstack(tiles), legend]), [cv2.IMWRITE_JPEG_QUALITY, 88])


def scatter3d(panels: list[tuple[str, np.ndarray]], out: Path):
    fig = plt.figure(figsize=(4.2 * len(panels), 5.2))
    for i, (title, p) in enumerate(panels):
        ax = fig.add_subplot(1, len(panels), i + 1, projection="3d")
        ok = np.isfinite(p).all(1)
        idx = np.arange(len(p))
        ax.plot(p[ok, 0], p[ok, 2], p[ok, 1], color="0.75", lw=0.4)
        ax.scatter(p[ok, 0], p[ok, 2], p[ok, 1], c=idx[ok], cmap="turbo", s=5)
        ax.set_title(title, fontsize=10)
        ax.set_box_aspect((np.ptp(p[ok, 0]), np.ptp(p[ok, 2]), np.ptp(p[ok, 1])))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=12, azim=-60)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def align_to(est: np.ndarray, truth: np.ndarray, reflect: bool) -> np.ndarray:
    from .metrics import align
    return align(est, truth, allow_reflection=reflect)[0]


def drift_plot(real_offsets: dict[str, list[tuple[int, int]]], shift_scan: dict[str, dict[int, float]], out: Path):
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    for name, pts in real_offsets.items():
        x, y = zip(*pts)
        axs[0].plot(x, y, marker="o", label=name)
    axs[0].axhline(0, color="k", lw=0.5)
    axs[0].set_xlabel("LED index (window start)")
    axs[0].set_ylabel("index offset vs tue1 (LEDs)")
    axs[0].set_title("Real captures: views drift apart along the string")
    axs[0].legend(fontsize=8)
    for name, sc in shift_scan.items():
        d = sorted(sc)
        axs[1].plot(d, [sc[k] for k in d], marker=".", label=name)
    axs[1].set_xlabel("index shift applied to second view")
    axs[1].set_ylabel("epipolar inlier fraction")
    axs[1].set_title("One shift fits synthetic views; none fit real ones")
    axs[1].legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def id_map(video: Path, plan, out: Path):
    """A phone frame, the decoded LED-ID map and the blob centroids."""
    from .video import DecodeParams, accumulate, _video_frames, blobs, decode_symbols, find_sync, \
        read_small, refine_sync_transitions
    dp = DecodeParams()
    series, ts, fps, (w, h), smalls, change = read_small(video, dp.downsample, dp.bands, keep=True)
    sync = refine_sync_transitions(change, ts, plan, find_sync(series, ts, plan, dp), series=series)
    acc, wsum = accumulate(_video_frames(video), ts, plan, sync, dp.guard_ms, 3 if plan.params.colour else 1)
    dec = decode_symbols(acc, wsum, plan, dp, plan.params.colour)
    dets = blobs(dec, plan.params.n_leds, dp)
    cap = cv2.VideoCapture(str(video))
    for _ in range(int(len(ts) * 0.6)):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    ids = dec["ids"]
    hue = (ids.astype(np.float32) * 179 / plan.params.n_leds).astype(np.uint8)
    hsv = np.stack([hue, np.full_like(hue, 255), np.where(ids >= 0, 255, 0).astype(np.uint8)], -1)
    idimg = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    pts = cv2.convertScaleAbs(frame, alpha=0.6)
    for led, d in dets.items():
        if d is not None and d.conf >= 0.5:
            cv2.circle(pts, (int(d.x), int(d.y)), 3, (0, 255, 0), -1)
    lab = [("phone frame (synthetic)", frame), ("decoded LED ID per pixel", idimg), ("confident detections", pts)]
    tiles = []
    for name, im in lab:
        im = im.copy()
        cv2.putText(im, name, (8, 22), 0, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(im)
    cv2.imwrite(str(out), np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 88])


def bar_chart(rows: list[tuple[str, float, str]], title: str, out: Path, xlabel: str = "median 3D error (cm)"):
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.2))
    names = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    cols = [r[2] for r in rows][::-1]
    ax.barh(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(v, i, f" {v:.1f}" if np.isfinite(v) else " failed", va="center", fontsize=8)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(out, dpi=110)
    plt.close(fig)


def load_json(p: Path):
    return json.loads(Path(p).read_text())
