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
from .render import (SimParams, Sensor, ambient_series, band_drive, make_view, radiance,
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
        bands = band_drive(view, rgb, starts, durs, t_tree % loop, params, loop_ms=loop)
        scene = radiance(view, drive, params, amb[k], band_drives=bands)
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
    min_margin: float = 0.35        # per-pixel decode margin, relative to the code's maximum
    min_amplitude_sigma: float = 6.0
    min_pixels: int = 3
    colour_box: int = 1             # colour mode: average RGB over this box first (video chroma is 4:2:0)


def _linear(frame_bgr: np.ndarray) -> np.ndarray:
    return SRGB_LUT[frame_bgr][..., ::-1]  # RGB, linear


def read_small(path: Path, ds: int, bands: int, keep: bool = False):
    """Downsampled linear frames and a per-band sync series built from LED pixels.

    The series uses only pixels that carry direct LED light: the top 2% by
    temporal variation of local contrast. Away from the LEDs the image follows
    the camera's exposure and noise, not the tree, and that biases the timing.
    """
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    ts, smalls = [], []
    h = w = None
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        ts.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        h, w = fr.shape[:2]
        small = cv2.resize(fr, (w // ds, h // ds), interpolation=cv2.INTER_AREA)
        smalls.append(_linear(small).astype(np.float32))
    cap.release()
    ts = np.array(ts)
    if len(ts) > 1 and (np.diff(ts) <= 0).any():
        ts = np.arange(len(ts)) * 1000 / fps
    # Local contrast (light above the local level, relative to that level) is
    # invariant to auto exposure, and static for light fittings and texture,
    # so only blinking LEDs vary in time.
    lum = np.stack([f.mean(-1) for f in smalls])            # (F, h, w)
    base = np.stack([cv2.GaussianBlur(l, (0, 0), 3) for l in lum])
    rel = np.maximum(lum - base, 0) / (base + 1e-4)
    var = rel.std(0)
    mask = var > np.percentile(var, 98)
    rows = np.array_split(np.arange(lum.shape[1]), bands)
    series = np.stack([np.where(mask[r][None], rel[:, r], 0).sum((1, 2)) / max(mask[r].sum(), 1)
                       for r in rows], 1)
    # Frame-to-frame change energy of LED pixels per band: every LED changes at
    # every pattern/complement boundary, so this carries the symbol clock.
    diff = np.abs(np.diff(rel, axis=0))
    change = np.stack([np.where(mask[r][None], diff[:, r], 0).sum((1, 2)) / max(mask[r].sum(), 1)
                       for r in rows], 1)
    if keep:
        return series, ts, fps, (w, h), smalls, change
    return series, ts, fps, (w, h)


def plan_brightness(plan: Plan) -> np.ndarray:
    """Expected relative scene brightness per tree frame."""
    p = plan.frame_power()
    return p / max(p.max(), 1)


def _sync_scores(y: np.ndarray, ts: np.ndarray, tmpl: np.ndarray, loop: float, phases: np.ndarray,
                 band_frac: np.ndarray, ro: float, ex: float) -> np.ndarray:
    """NCC between the (F, B) series and the template for each candidate phase."""
    k = max(1, int(round(ex)))
    box = np.convolve(np.concatenate([tmpl, tmpl[:k]]), np.ones(k) / k, "valid")[:len(tmpl)]
    t = ts[:, None] + band_frac[None, :] * ro
    tt = np.round(t[None] + phases[:, None, None]).astype(int) % int(loop)
    pred = box[tt]
    pred = pred - pred.mean(1, keepdims=True)
    pred /= pred.std(1, keepdims=True) + 1e-9
    return (pred * y[None]).mean((1, 2))


def find_sync(series: np.ndarray, ts: np.ndarray, plan: Plan, dp: DecodeParams) -> dict:
    """Loop phase, rolling-shutter readout and exposure from the sparkle series.

    Stage 1 uses the whole frame (one band), which measures the frame-centre
    time robustly: phase + readout / 2. Stage 2 splits the frame into bands and
    fits only the readout, with the phase tied to the stage 1 estimate.
    """
    loop = plan.loop_ms
    bright = plan_brightness(plan)
    starts = plan.starts_ms()
    grid = np.arange(int(loop))
    tmpl = bright[np.searchsorted(starts, grid, side="right") - 1]
    nb = series.shape[1]

    def norm(x):
        x = x - x.mean(0, keepdims=True)
        return x / (x.std(0, keepdims=True) + 1e-9)

    whole = norm(series.mean(1, keepdims=True))
    # Coarse phase step scales with the loop (1 ms for the standard 6.6 s loop,
    # coarser for the 54 s one-LED-at-a-time loop), then a local 1 ms search.
    step = max(1.0, float(np.floor(loop / 6600)))
    best = None
    for ex in dp.exposures_ms:
        phases = np.arange(0, loop, step)
        sc = _sync_scores(whole, ts, tmpl, loop, phases, np.array([0.5]), 0.0, ex)
        j = int(np.argmax(sc))
        if step > 1:
            fine = (phases[j] + np.arange(-step, step + 1, 1.0)) % loop
            scf = _sync_scores(whole, ts, tmpl, loop, fine, np.array([0.5]), 0.0, ex)
            k = int(np.argmax(scf))
            cand = (float(scf[k]), float(fine[k]), ex)
        else:
            cand = (float(sc[j]), float(phases[j]), ex)
        if best is None or cand[0] > best[0]:
            best = cand
    score1, centre, ex = best
    # centre is the phase that puts the frame centre (band 0.5) on the template;
    # with readout ro the row-0 phase is centre - ro / 2.
    y = norm(series)
    band_frac = (np.arange(nb) + 0.5) / nb
    best2 = None
    for ro in np.arange(0, 41, 5.0):
        ph = centre - 0.5 * ro + np.arange(-6, 7, 1.0)
        sc = _sync_scores(y, ts, tmpl, loop, ph % loop, band_frac, ro, ex)
        j = int(np.argmax(sc))
        if best2 is None or sc[j] > best2[0]:
            best2 = (float(sc[j]), float(ph[j] % loop), float(ro))
    return {"score": score1, "band_score": best2[0], "phase_ms": best2[1], "readout_ms": best2[2],
            "exposure_ms": float(ex)}


def accumulate(frames, ts: np.ndarray, plan: Plan, sync: dict, guard_ms: float, channels: int):
    """Average each tree frame's image over the clean rows of all video frames.

    A row of a video frame is clean for tree frame j if its exposure window,
    [t + row / H * readout, + exposure], lies inside frame j shrunk by the guard.
    """
    loop = plan.loop_ms
    starts = plan.starts_ms().astype(float)
    ends = starts + plan.duration_ms
    ro, ex, ph = sync["readout_ms"], sync["exposure_ms"], sync["phase_ms"]
    acc = wsum = None
    for k, lin in enumerate(frames):
        if k >= len(ts):
            break
        if channels == 1 and lin.shape[-1] != 1:
            lin = lin.mean(-1, keepdims=True)
        H = lin.shape[0]
        if acc is None:
            acc = np.zeros((plan.n_frames, H, lin.shape[1], channels), np.float32)
            wsum = np.zeros((plan.n_frames, H), np.float32)
        r_start = ts[k] + ph + np.arange(H) / H * ro
        r_end = r_start + ex
        loop_idx = np.floor(r_start / loop)
        for L in np.unique(loop_idx):
            s_loc = r_start - L * loop
            e_loc = r_end - L * loop
            for j in np.nonzero((ends > s_loc.min() - 1) & (starts < e_loc.max() + 1))[0]:
                good = (s_loc >= starts[j] + guard_ms) & (e_loc <= ends[j] - guard_ms)
                if good.any():
                    acc[j, good] += lin[good]
                    wsum[j, good] += 1
    return acc, wsum


def _video_frames(path: Path):
    cap = cv2.VideoCapture(str(path))
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        yield _linear(fr)
    cap.release()


def symbol_images(path: Path, plan: Plan, sync: dict, dp: DecodeParams, ts: np.ndarray,
                  channels: int) -> tuple[np.ndarray, np.ndarray]:
    return accumulate(_video_frames(path), ts, plan, sync, dp.guard_ms, channels)


def decode_quality(acc: np.ndarray, wsum: np.ndarray, plan: Plan, colour: bool) -> float:
    """Amplitude-weighted mean decode margin: how cleanly the symbols decode."""
    if (wsum[[f for f in range(plan.n_frames) if plan.roles[f] in ("code", "off")]] == 0).mean() > 0.5:
        return 0.0
    accn = acc / np.maximum(wsum[..., None, None], 1e-6)
    off = accn[plan.index("off")[0]]
    amp = np.zeros(off.shape[:2], np.float32)
    for f in range(plan.n_frames):
        if plan.roles[f] == "code":
            amp = np.maximum(amp, np.abs(accn[f] - off).max(-1))
    thr = np.percentile(amp, 97)
    pix = np.nonzero(amp > thr)
    if len(pix[0]) < 10:
        return 0.0
    sb, a = _soft_bits(accn, plan, pix, colour)
    W = plan.codebook.words.astype(np.float32) * 2 - 1
    score = sb @ W.T / plan.codebook.n_bits
    part = -np.partition(-score, 1, axis=1)
    cb = plan.codebook
    margin = (part[:, 0] - part[:, 1]) / (2 * cb.min_distance / cb.n_bits)
    return float(np.average(np.clip(margin, 0, 1), weights=a))


def refine_sync_transitions(change: np.ndarray, ts: np.ndarray, plan: Plan, sync: dict,
                            span_ms: float = 64.0, n_sample: int = 120, seed: int = 0,
                            series: np.ndarray | None = None) -> dict:
    """Refine phase, readout and exposure by matching frame-to-frame change energy.

    For each candidate, predict for every band the mean |change| of a sample
    of LEDs' exposure-integrated drive between consecutive video frames and
    correlate it with the measured change energy of LED pixels. That template
    is nearly periodic in the symbol length, so the brightness correlation
    (precise to tens of ms but unambiguous) is added to pick the right symbol.
    """
    from .render import _window_drive
    rng = np.random.default_rng(seed)
    rgb = plan.rgb().mean(-1)                        # (F_tree, n) mono drive
    leds = rng.choice(rgb.shape[1], min(n_sample, rgb.shape[1]), replace=False)
    rgb = rgb[:, leds][..., None]
    loop = plan.loop_ms
    starts = plan.starts_ms().astype(float)
    durs = plan.duration_ms.astype(float)
    B = change.shape[1]
    frac = (np.arange(B) + 0.5) / B
    valid = change.std(0) > 0
    y = change[:, valid]
    y = (y - y.mean(0)) / (y.std(0) + 1e-9)
    bright_tmpl = plan_brightness(plan)[np.searchsorted(starts, np.arange(int(loop)), side="right") - 1]
    whole = None
    if series is not None:
        whole = series.mean(1, keepdims=True)
        whole = (whole - whole.mean()) / (whole.std() + 1e-9)
    def evaluate(centre, dphs, exs, ros):
        best = None
        for ex in exs:
            for ro in ros:
                bsc = np.zeros(len(dphs))
                if whole is not None:
                    bsc = _sync_scores(whole, ts, bright_tmpl, loop, (centre + dphs + 0.5 * ro) % loop,
                                       np.array([0.0]), 0.0, ex)
                for dph, b_score in zip(dphs, bsc):
                    ph = centre + dph
                    t = (ts[:, None] + ph + frac[None, valid] * ro) % loop       # (F, Bv)
                    d = _window_drive(rgb, starts, durs, t.ravel(), ex, loop)[..., 0]
                    d = d.reshape(t.shape + (len(leds),))
                    pred = np.abs(np.diff(d, axis=0)).mean(-1)                    # (F-1, Bv)
                    pred = (pred - pred.mean(0)) / (pred.std(0) + 1e-9)
                    sc = float((pred * y).mean()) + float(b_score)
                    if best is None or sc > best[0]:
                        best = (sc, ph % loop, ro, ex)
        return best

    # Coarse over the full span on a reduced grid, then fine around the best.
    coarse = evaluate(sync["phase_ms"], np.arange(-span_ms, span_ms + 1, 4.0), (4.0, 12.0, 24.0),
                      (0.0, 10.0, 20.0, 30.0, 40.0))
    best = evaluate(coarse[1], np.arange(-6, 7, 1.0), (2.0, 8.0, 16.0, 33.0), (0.0, 10.0, 20.0, 30.0, 40.0))
    out = dict(sync)
    out.update({"phase_ms": float(best[1]), "readout_ms": best[2], "exposure_ms": best[3],
                "transition_score": best[0]})
    return out


def refine_sync(smalls: list, ts: np.ndarray, plan: Plan, sync: dict, dp: DecodeParams,
                colour: bool) -> dict:
    """Coordinate search on phase, readout and exposure maximising decode quality."""
    channels = 3 if colour else 1
    frames = smalls if channels == 3 else [f.mean(-1, keepdims=True) for f in smalls]
    best = dict(sync)

    def q(c):
        acc, wsum = accumulate(frames, ts, plan, c, dp.guard_ms, channels)
        return decode_quality(acc, wsum, plan, colour)

    best["quality"] = q(best)
    for _round in range(2):
        for key, grid in (("phase_ms", best["phase_ms"] + np.arange(-36, 37, 4.0)),
                          ("readout_ms", np.arange(0, 41, 5.0)),
                          ("exposure_ms", np.array([2.0, 4.0, 8.0, 16.0, 24.0, 33.0])),
                          ("phase_ms", best["phase_ms"] + np.arange(-4, 5, 1.0))):
            for val in grid:
                cand = dict(best)
                cand[key] = float(val % plan.loop_ms) if key == "phase_ms" else float(val)
                qq = q(cand)
                if qq > best["quality"] + 1e-4:
                    best, best["quality"] = cand, qq
    return best


def _soft_bits(acc: np.ndarray, plan: Plan, pix: tuple, colour: bool):
    """Soft bits (Npix, n_bits) in [-1, 1] and the amplitude per pixel.

    Colour mode unmixes camera RGB into LED channels with one global matrix
    (crosstalk is a property of the camera and the LED type), estimated from
    the R, G and B training frames over bright pixels. Inverting each pixel's
    own noisy 3x3 matrix amplifies noise. Soft bits are normalised per LED
    channel, so a channel that is weak at a pixel still gets full weight.
    """
    nb = plan.codebook.n_bits
    off = acc[plan.index("off")[0]][pix]  # (N, C)
    chan_of_bit = np.zeros(nb, int)
    if colour:
        Mp = np.stack([acc[plan.index(c)[0]][pix] - off for c in "rgb"], -1)  # (N, cam, led)
        strength = np.abs(Mp).sum((1, 2))
        bright = strength >= np.percentile(strength, 50)
        Mn = Mp[bright] / np.maximum(np.abs(Mp[bright]).sum(1, keepdims=True), 1e-9)  # column-normalised
        Mg = np.median(Mn, axis=0)
        Mg_inv = np.linalg.pinv(Mg)
    pos, neg = {}, {}
    for f, bits in enumerate(plan.bits):
        if plan.roles[f] != "code":
            continue
        img = acc[f][pix] - off
        if colour:
            x = img @ Mg_inv.T  # LED channel intensities (N, 3)
            for c, (bit, pol) in enumerate(bits):
                (pos if pol else neg)[bit] = x[:, c]
                chan_of_bit[bit] = c
        else:
            (bit, pol), = bits
            (pos if pol else neg)[bit] = img[:, 0]
    s = np.zeros((len(off), nb), np.float32)
    if neg:
        for bit in range(nb):
            s[:, bit] = pos[bit] - neg[bit]
    else:
        # No complement frames: threshold each pixel at the midpoint of its own
        # brightest and darkest code frames (the all-on reference is a poor
        # threshold: it includes every neighbour's light and a different
        # exposure). An LED whose codeword is all zeros or all ones cannot be
        # decoded this way.
        P = np.stack([pos[bit] for bit in range(nb)], 1)
        mid = 0.5 * (P.max(1, keepdims=True) + P.min(1, keepdims=True))
        s = (P - mid).astype(np.float32)
    amp = np.zeros(len(off), np.float32)
    for c in np.unique(chan_of_bit):
        cols = chan_of_bit == c
        a_c = np.abs(s[:, cols]).max(1)
        s[:, cols] /= np.maximum(a_c[:, None], 1e-9)
        amp = np.maximum(amp, a_c)
    return np.clip(s, -1, 1), amp


def decode_symbols(acc: np.ndarray, wsum: np.ndarray, plan: Plan, dp: DecodeParams,
                   colour: bool) -> dict:
    n = plan.params.n_leds
    acc = acc / np.maximum(wsum[..., None, None], 1e-6)
    if colour and dp.colour_box > 1:
        # Phone and H.264 video store colour at half resolution and quantise it
        # coarsely, so per-pixel colour around a 2-3 px LED core is unreliable.
        k = (dp.colour_box, dp.colour_box)
        acc = np.stack([np.stack([cv2.blur(acc[f][..., c], k) for c in range(acc.shape[-1])], -1)
                        for f in range(acc.shape[0])])
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
    # The best-vs-second gap can be at most 2 d / n for a code of minimum
    # distance d, so normalise it to [0, 1] to compare codes fairly.
    cb = plan.codebook
    margin = np.abs(s1 - s2) / (2 * cb.min_distance / cb.n_bits)
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
        conf = float(np.clip(mm * 1.2, 0, 1) * (1 - min(second, 1)) * min(1.0, sel.sum() / 6))
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
    series, ts, fps, (w, h), smalls, change = read_small(path, dp.downsample, dp.bands, keep=True)
    colour = plan.params.colour
    sync = find_sync(series, ts, plan, dp)
    if not plan.roles.count("single"):
        sync = refine_sync_transitions(change, ts, plan, sync, series=series)
    del smalls
    if verbose:
        print(f"  {path.name}: {len(ts)} frames @ {fps:.1f} fps, sync {sync}")
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
