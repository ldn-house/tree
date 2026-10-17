"""Video calibration: simulate a phone filming the tree, and decode such videos.

The tree loops a sequence (calib.sequence). Each view is a separate clip that
starts at an arbitrary point in the loop. The decoder needs no shared clock:

1. Sync. A downsampled brightness series per horizontal band is correlated
   against the brightness the plan predicts, over loop phase, rolling-shutter
   readout time and exposure. That gives the loop start to about a millisecond.
2. Symbol images. For every symbol occurrence, each image row is taken only
   from frames whose exposure window for that row lies inside the symbol
   (minus a guard), averaged over loops.
3. Per-pixel decoding. Soft bits come from pattern minus complement,
   normalised by the pixel's own amplitude (after per-pixel colour unmixing in
   colour mode), correlated against every codeword; the margin between the
   best and second-best codeword is the confidence.
4. Blobs. Pixels are grouped by decoded ID; the strongest connected component
   is kept and its core gives a sub-pixel centroid.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .camera import Camera
from .detect import SRGB_LUT, Detection
from .layout import Layout
from .render import (SimParams, Sensor, ambient_series, make_view, radiance,
                     rolling_shutter_drive)
from .sequence import Plan, SequenceParams, build, sequential_plan

# ---------------------------------------------------------------- simulation


@dataclass
class VideoSpec:
    fps: float = 30.0
    duration_s: float | None = None   # default: 1.6 loops
    start_ms: float | None = None     # tree time at video start (default random)
    clock_ppm: float = 40.0           # tree clock vs camera clock
    jitter_ms: float = 1.0            # frame timestamp jitter
    crf: int = 20                     # x264 quality (phones record ~10-20 Mbit/s)
    scale: float = 1.0
    awb: bool = False                 # auto white balance reacting to the scene


def _ffmpeg_writer(path: Path, w: int, h: int, fps: float, crf: int):
    return subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
         "-r", f"{fps}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
         "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)


def render_video(layout: Layout, cam: Camera, plan: Plan, params: SimParams, spec: VideoSpec,
                 out: Path, seed: int = 0) -> dict:
    """Render one clip of the looping plan. Returns truth for this view."""
    rng = np.random.default_rng(seed)
    view = make_view(layout, cam, params, np.random.default_rng(seed + 1))
    sensor = Sensor(params, np.random.default_rng(seed + 2))
    loop = plan.loop_ms
    dur = spec.duration_s if spec.duration_s else 1.6 * loop / 1000
    n_frames = int(dur * spec.fps)
    start = rng.uniform(0, loop) if spec.start_ms is None else spec.start_ms
    rgb = plan.rgb()
    starts = plan.starts_ms().astype(float)
    durs = plan.duration_ms.astype(float)
    amb = ambient_series(params, n_frames, rng)
    writer = _ffmpeg_writer(out, cam.width, cam.height, spec.fps, spec.crf)
    times = []
    wb = np.array(params.wb, np.float32)
    for k in range(n_frames):
        tau = k * 1000 / spec.fps + rng.normal(0, spec.jitter_ms)
        t_tree = start + tau * (1 + spec.clock_ppm * 1e-6)
        times.append(t_tree)
        drive = rolling_shutter_drive(view, rgb, starts, durs, t_tree % loop, params, loop_ms=loop)
        scene = radiance(view, drive, params, amb[k])
        if spec.awb:
            # Grey-world AWB with lag: gains drift towards making the scene grey.
            mean = scene[::8, ::8].reshape(-1, 3).mean(0) * wb + 1e-6
            target = wb * (mean.mean() / mean)
            wb = wb * (target / wb) ** 0.15
            params.wb = tuple(wb)
        img = sensor.expose(scene, settle=1)
        writer.stdin.write(img.tobytes())
    writer.stdin.close()
    writer.wait()
    return {"camera": cam.to_dict(), "uv": np.nan_to_num(view.uv, nan=-1).tolist(),
            "visible": view.visible.tolist(), "start_ms": start, "fps": spec.fps,
            "frame_tree_ms": times, "loop_ms": loop}


def _render_video_job(job):
    layout, cam, plan_params, params, spec, out, seed, sequential = job
    plan = sequential_plan(symbol_ms=plan_params.symbol_ms) if sequential else build(plan_params)
    return render_video(layout, cam, plan, params, spec, out, seed)


def simulate_videos(layout: Layout, cams: list[Camera], plan_params: SequenceParams, params: SimParams,
                    spec: VideoSpec, out: Path, seed: int = 0, sequential: bool = False,
                    workers: int = 4) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(layout, cam, plan_params, params, spec, out / f"{cam.name}.mp4", seed * 100 + i, sequential)
            for i, cam in enumerate(cams)]
    with ProcessPoolExecutor(workers) as ex:
        views = list(ex.map(_render_video_job, jobs))
    truth = {"layout": layout.name, "pitch": layout.pitch, "points": layout.points.tolist(),
             "plan": vars(plan_params) | {"sequential": sequential}, "sim": _jsonable(vars(params)),
             "video": _jsonable(vars(spec)), "views": views}
    (out / "truth.json").write_text(json.dumps(truth))
    return truth


def _jsonable(d):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in d.items()}


# ---------------------------------------------------------------- decoding


@dataclass
class DecodeParams:
    downsample: int = 4
    bands: int = 8
    guard_ms: float = 4.0
    readouts_ms: tuple = (0.0, 10.0, 20.0, 30.0)
    exposures_ms: tuple = (2.0, 8.0, 16.0, 30.0)
    min_margin: float = 0.25        # per-pixel decode margin (0..1) to keep a pixel
    min_amplitude_sigma: float = 6.0
    min_pixels: int = 3


def _linear(frame_bgr: np.ndarray) -> np.ndarray:
    return SRGB_LUT[frame_bgr][..., ::-1]  # RGB, linear


def read_small(path: Path, ds: int, bands: int):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    series, ts = [], []
    h = w = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w = fr.shape[:2]
        small = cv2.resize(fr, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        lum = _linear(small).mean(-1)
        # Sparkle: light above the local level, relative to that level. Auto
        # exposure scales the whole image, so the ratio ignores it; the room and
        # foliage are smooth, the LEDs are points.
        base = cv2.GaussianBlur(lum, (0, 0), 3)
        sparkle = np.maximum(lum - base, 0)
        series.append([sp.sum() / (bs.sum() + 1e-6) for sp, bs in
                       zip(np.array_split(sparkle, bands, axis=0), np.array_split(base, bands, axis=0))])
    cap.release()
    ts = np.array(ts)
    if len(ts) > 1 and (np.diff(ts) <= 0).any():
        ts = np.arange(len(ts)) * 1000 / fps
    return np.array(series), ts, fps, (w, h)


def plan_brightness(plan: Plan) -> np.ndarray:
    """Expected relative scene brightness per tree frame."""
    p = plan.frame_power()
    return p / max(p.max(), 1)


def find_sync(series: np.ndarray, ts: np.ndarray, plan: Plan, dp: DecodeParams) -> dict:
    """Loop phase, readout and exposure that best explain the brightness series."""
    loop = plan.loop_ms
    bright = plan_brightness(plan)
    starts = plan.starts_ms()
    # Template sampled at 1 ms over one loop.
    grid = np.arange(int(loop))
    idx = np.searchsorted(starts, grid, side="right") - 1
    tmpl = bright[idx]
    n_frames, nb = series.shape
    y = series - series.mean(0, keepdims=True)
    y /= y.std(0, keepdims=True) + 1e-9
    band_frac = (np.arange(nb) + 0.5) / nb
    best = None
    phases = np.arange(0, loop, 1.0)
    for ro in dp.readouts_ms:
        for ex in dp.exposures_ms:
            k = max(1, int(ex))
            box = np.convolve(np.concatenate([tmpl, tmpl[:k]]), np.ones(k) / k, "valid")[:len(tmpl)]
            # Sample time for frame i, band b, phase p: (ts[i] + band*ro + p) mod loop
            t = ts[:, None] + band_frac[None, :] * ro  # (F, B)
            tt = (t[None] + phases[:, None, None]).astype(int) % int(loop)
            pred = box[tt]  # (P, F, B)
            pred = pred - pred.mean(1, keepdims=True)
            pred /= pred.std(1, keepdims=True) + 1e-9
            score = (pred * y[None]).mean((1, 2))
            j = int(np.argmax(score))
            if best is None or score[j] > best["score"]:
                best = {"score": float(score[j]), "phase_ms": float(phases[j]), "readout_ms": ro,
                        "exposure_ms": ex}
    # Refine phase to 0.25 ms around the best.
    return best


def symbol_images(path: Path, plan: Plan, sync: dict, dp: DecodeParams, ts: np.ndarray,
                  channels: int) -> tuple[np.ndarray, np.ndarray]:
    """Average each tree frame's image over all clean rows of all frames and loops."""
    cap = cv2.VideoCapture(str(path))
    loop = plan.loop_ms
    starts = plan.starts_ms().astype(float)
    ends = starts + plan.duration_ms
    ro, ex, ph = sync["readout_ms"], sync["exposure_ms"], sync["phase_ms"]
    acc = None
    wsum = None
    k = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if k >= len(ts):
            break
        lin = _linear(fr)
        if channels == 1:
            lin = lin.mean(-1, keepdims=True)
        H = lin.shape[0]
        if acc is None:
            acc = np.zeros((plan.n_frames, H, lin.shape[1], channels), np.float32)
            wsum = np.zeros((plan.n_frames, H), np.float32)
        t0 = (ts[k] + ph)  # tree time (unwrapped within this clip) of row 0 exposure start
        rows = np.arange(H)
        r_start = t0 + rows / H * ro
        r_end = r_start + ex
        loop_idx = np.floor(r_start / loop)
        for L in np.unique(loop_idx):
            s_loc = r_start - L * loop
            e_loc = r_end - L * loop
            for j in np.nonzero((ends > s_loc.min() - 1) & (starts < e_loc.max() + 1))[0]:
                good = (s_loc >= starts[j] + dp.guard_ms) & (e_loc <= ends[j] - dp.guard_ms)
                if good.any():
                    acc[j, good] += lin[good]
                    wsum[j, good] += 1
        k += 1
    cap.release()
    return acc, wsum


def _soft_bits(acc: np.ndarray, plan: Plan, pix: tuple, colour: bool):
    """Soft bits (Npix, n_bits) in [-1, 1] and the amplitude per pixel."""
    nb = plan.codebook.n_bits
    off = acc[plan.index("off")[0]][pix]  # (N, C)
    if colour:
        M = np.stack([acc[plan.index(c)[0]][pix] - off for c in "rgb"], -1)  # (N, 3 cam, 3 led)
        Minv = np.linalg.pinv(M)
    pos = {}
    neg = {}
    for f, b in enumerate(plan.bits):
        if plan.roles[f] != "code":
            continue
        img = acc[f][pix] - off
        if colour:
            x = np.einsum("nij,nj->ni", Minv, img)  # LED channel intensities
            for c, (bit, pol) in enumerate(b):
                (pos if pol else neg)[bit] = x[:, c]
        else:
            (bit, pol), = b
            (pos if pol else neg)[bit] = img[:, 0]
    s = np.zeros((len(off), nb), np.float32)
    amp = np.zeros(len(off), np.float32)
    for bit in range(nb):
        if bit in neg:
            d = pos[bit] - neg[bit]
        else:
            # No complement: threshold at half the "on" reference, rescaled from
            # its level to the code-frame level.
            f_on = plan.index("on")[0]
            f_code = next(f for f in range(plan.n_frames) if plan.roles[f] == "code")
            ratio = plan.level[f_code].max() / max(plan.level[f_on].max(), 1)
            d = pos[bit] - 0.5 * ratio * np.maximum(acc[f_on][pix].mean(-1) - off.mean(-1), 1e-6)
        s[:, bit] = d
        amp = np.maximum(amp, np.abs(d))
    s /= np.maximum(amp[:, None], 1e-9)
    return np.clip(s, -1, 1), amp


def decode_symbols(acc: np.ndarray, wsum: np.ndarray, plan: Plan, dp: DecodeParams,
                   colour: bool) -> dict:
    n = plan.params.n_leds
    acc = acc / np.maximum(wsum[..., None, None], 1e-6)
    off = acc[plan.index("off")[0]]
    code_frames = [f for f in range(plan.n_frames) if plan.roles[f] == "code"]
    # Amplitude map from the strongest pattern/complement difference.
    amp = np.zeros(off.shape[:2], np.float32)
    for f in code_frames:
        amp = np.maximum(amp, np.abs(acc[f] - off).max(-1))
    sigma = 1.4826 * np.median(np.abs(amp - np.median(amp))) + 1e-9
    mask = amp > np.median(amp) + dp.min_amplitude_sigma * sigma
    pix = np.nonzero(mask)
    s, a = _soft_bits(acc, plan, pix, colour)
    W = plan.codebook.words.astype(np.float32) * 2 - 1  # (n, bits)
    score = s @ W.T / plan.codebook.n_bits
    top2 = np.argpartition(-score, 1, axis=1)[:, :2]
    s1 = score[np.arange(len(score)), top2[:, 0]]
    s2 = score[np.arange(len(score)), top2[:, 1]]
    first = np.where(s1 >= s2, top2[:, 0], top2[:, 1])
    margin = np.abs(s1 - s2)
    best = np.maximum(s1, s2)
    keep = (margin >= dp.min_margin) & (best > 0.5)
    ids = np.full(off.shape[:2], -1, np.int32)
    ids[pix[0][keep], pix[1][keep]] = first[keep]
    amp_img = np.zeros_like(amp)
    amp_img[pix] = a
    marg_img = np.zeros_like(amp)
    marg_img[pix] = margin
    return {"ids": ids, "amp": amp_img, "margin": marg_img, "mask": mask}


def blobs(dec: dict, n: int, dp: DecodeParams) -> dict[int, Detection | None]:
    ids, amp, marg = dec["ids"], dec["amp"], dec["margin"]
    ys, xs = np.nonzero(ids >= 0)
    lab = ids[ys, xs]
    order = np.argsort(lab, kind="stable")
    ys, xs, lab = ys[order], xs[order], lab[order]
    cuts = np.searchsorted(lab, np.arange(n + 1))
    out: dict[int, Detection | None] = {}
    for led in range(n):
        a, b = cuts[led], cuts[led + 1]
        if b - a < dp.min_pixels:
            out[led] = None
            continue
        y, x = ys[a:b], xs[a:b]
        w = amp[y, x] * marg[y, x]
        x0, y0 = x.min(), y.min()
        m = np.zeros((y.max() - y0 + 1, x.max() - x0 + 1), np.uint8)
        m[y - y0, x - x0] = 1
        ncomp, comp = cv2.connectedComponents(cv2.dilate(m, np.ones((3, 3), np.uint8)), connectivity=8)
        cid = comp[y - y0, x - x0]
        tot = np.bincount(cid, weights=w, minlength=ncomp)
        order_c = np.argsort(-tot)
        main = order_c[0]
        second = tot[order_c[1]] / tot[main] if ncomp > 2 and len(order_c) > 1 and tot[order_c[1]] > 0 else 0.0
        sel = cid == main
        ya, xa, wa, aa = y[sel], x[sel], w[sel], amp[y[sel], x[sel]]
        core = aa >= 0.5 * aa.max()
        wc = (aa[core] - 0.5 * aa.max()) + 1e-9
        cx = float((xa[core] * wc).sum() / wc.sum())
        cy = float((ya[core] * wc).sum() / wc.sum())
        mm = float(np.average(marg[ya, xa], weights=wa))
        conf = float(np.clip(mm * 1.5, 0, 1) * (1 - min(second, 1)) * min(1.0, sel.sum() / 6))
        out[led] = Detection(cx + 0.0, cy + 0.0, float(aa.max()), float(np.log10(sel.sum())), mm, float(second),
                             0.0, conf)
    return out


def decode_sequential(acc: np.ndarray, wsum: np.ndarray, plan: Plan) -> dict[int, Detection | None]:
    from .detect import locate
    accn = acc / np.maximum(wsum[..., None, None], 1e-6)
    lum = accn.mean(-1)
    singles = plan.index("single")
    out = {}
    for k, f in enumerate(singles):
        neigh = [lum[g] for g in (singles[k - 1] if k else None, singles[k + 1] if k + 1 < len(singles) else None)
                 if g is not None]
        res = lum[f] - np.mean(neigh, axis=0)
        d = locate(res)
        if d is not None:
            from .detect import confidence, load_conf_model
            d.conf = confidence(d, load_conf_model())
        out[plan.bits[f][0]] = d
    return out


def decode_video(path: Path, plan: Plan, dp: DecodeParams | None = None, verbose: bool = False) -> dict:
    dp = dp or DecodeParams()
    series, ts, fps, (w, h) = read_small(path, dp.downsample, dp.bands)
    sync = find_sync(series, ts, plan, dp)
    if verbose:
        print(f"  {path.name}: {len(ts)} frames @ {fps:.1f} fps, sync {sync}")
    colour = plan.params.colour
    channels = 3 if colour else 1
    acc, wsum = symbol_images(path, plan, sync, dp, ts, channels)
    missing = [i for i in range(plan.n_frames) if wsum[i].max() == 0]
    if any(plan.roles[i] in ("code", "off", "on", "r", "g", "b", "single") for i in missing):
        sync["missing_symbols"] = len(missing)
    if plan.roles.count("single"):
        dets = decode_sequential(acc, wsum, plan)
    else:
        dec = decode_symbols(acc, wsum, plan, dp, colour)
        dets = blobs(dec, plan.params.n_leds, dp)
    return {"method": f"video:{plan.params.key()}", "lag": 0, "image_size": [w, h],
            "detections": dets, "sync": sync}
