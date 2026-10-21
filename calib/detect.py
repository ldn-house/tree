"""LED detection in still captures (one LED lit per frame).

Methods, from the existing analyze.py to the new ones:

    local_contrast   analyze.find_led_local_contrast (current default)
    background       analyze.find_blob_center against background.jpg
    ref              subtract a gain/offset-fitted all-off reference built from
                     the per-pixel median of the frames themselves
    diff             subtract the gain-fitted mean of the previous and next frames
    ref_dog          ref plus a difference-of-Gaussians band-pass that favours
                     compact LED cores over wide needle glow and wall bloom

The new methods work on linearised intensity, refine the peak to sub-pixel
precision with a thresholded weighted centroid, and report features from which
a confidence is computed. A low confidence is reported as an explicit miss
rather than the brightest reflection.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

SRGB_LUT = np.array([((i / 255) / 12.92) if i / 255 <= 0.04045 else (((i / 255) + 0.055) / 1.055) ** 2.4
                     for i in range(256)], np.float32)

FEATURES = ("log_snr", "compact", "second", "sat")


def linear_gray(img_bgr: np.ndarray) -> np.ndarray:
    lin = SRGB_LUT[img_bgr]
    if lin.ndim == 3:
        lin = lin @ np.array([0.114, 0.587, 0.299], np.float32)  # BGR luminance weights
    return lin.astype(np.float32)


@dataclass
class Detection:
    x: float
    y: float
    peak: float
    log_snr: float
    compact: float
    second: float
    sat: float
    conf: float = 1.0

    def features(self) -> np.ndarray:
        return np.array([getattr(self, f) for f in FEATURES])


def fit_gain(frame: np.ndarray, ref: np.ndarray, step: int = 4) -> tuple[float, float]:
    """Robust frame ~= a * ref + b over background pixels."""
    f = frame[::step, ::step].ravel()
    r = ref[::step, ::step].ravel()
    keep = np.ones(len(f), bool)
    a, b = 1.0, 0.0
    for _ in range(3):
        A = np.vstack([r[keep], np.ones(keep.sum())]).T
        a, b = np.linalg.lstsq(A, f[keep], rcond=None)[0]
        res = f - (a * r + b)
        s = 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + 1e-6
        keep = np.abs(res) < 3 * s
    return float(a), float(b)


def _noise_sigma(img: np.ndarray) -> float:
    s = img[::3, ::3]
    return float(1.4826 * np.median(np.abs(s - np.median(s))) + 1e-7)


def locate(residual: np.ndarray, score: np.ndarray | None = None, raw: np.ndarray | None = None,
           edge: int = 6, core_px: float = 4.0, exclude_px: float = 20.0) -> Detection | None:
    """Find the strongest compact peak and measure it.

    residual: LED-only image (linear). score: map to find the peak in (defaults to
    a lightly blurred residual). raw: original frame for saturation checks.
    """
    h, w = residual.shape
    if score is None:
        score = cv2.GaussianBlur(residual, (0, 0), 1.5)
    sc = score.copy()
    sc[:edge] = sc[-edge:] = 0
    sc[:, :edge] = sc[:, -edge:] = 0
    _, pmax, _, (px, py) = cv2.minMaxLoc(sc)
    if pmax <= 0:
        return None
    sigma = _noise_sigma(score)
    # Second peak: strongest point outside the exclusion radius.
    yy, xx = np.ogrid[:h, :w]
    far = (xx - px) ** 2 + (yy - py) ** 2 > exclude_px ** 2
    second = float(sc[far].max() / pmax) if far.any() else 0.0
    # Sub-pixel: weighted centroid of residual above half max within core radius.
    r = int(np.ceil(core_px * 2))
    x0, x1, y0, y1 = max(px - r, 0), min(px + r + 1, w), max(py - r, 0), min(py + r + 1, h)
    win = residual[y0:y1, x0:x1].astype(np.float64)
    wy, wx = np.mgrid[y0:y1, x0:x1]
    local_bg = np.median(residual[max(py - 3 * r, 0):min(py + 3 * r, h), max(px - 3 * r, 0):min(px + 3 * r, w)])
    wv = np.clip(win - local_bg, 0, None)
    thr = 0.5 * wv.max()
    wv = np.where(wv >= thr, wv - thr, 0)
    if wv.sum() <= 0:
        cx, cy = float(px), float(py)
    else:
        cx = float((wv * wx).sum() / wv.sum())
        cy = float((wv * wy).sum() / wv.sum())
    # Compactness: residual energy within core radius vs within 5x core radius.
    pos = np.clip(residual, 0, None)
    rr = int(core_px * 5)
    sub = (slice(max(int(cy) - rr, 0), min(int(cy) + rr + 1, h)), slice(max(int(cx) - rr, 0), min(int(cx) + rr + 1, w)))
    p_sub = pos[sub]
    sy, sx = np.mgrid[sub[0], sub[1]]
    d2 = (sx - cx) ** 2 + (sy - cy) ** 2
    inner = p_sub[d2 <= core_px ** 2].sum()
    outer = p_sub.sum() + 1e-9
    sat = 0.0
    if raw is not None:
        sat = float((raw[max(int(cy) - 2, 0):int(cy) + 3, max(int(cx) - 2, 0):int(cx) + 3] > 0.98).mean())
    return Detection(cx, cy, float(pmax), float(np.log10(max(pmax / sigma, 1e-3))), float(inner / outer),
                     second, sat)


class StillDetector:
    """Detect the single lit LED in each frame of a sequential capture."""

    def __init__(self, method: str = "ref_dog", ref_stride: int = 5):
        self.method = method
        self.ref_stride = ref_stride
        self.ref = None
        self.bg = None

    def prepare(self, frames: list[Path], background: Path | None = None):
        if self.method in ("ref", "ref_dog", "diff"):
            sample = [linear_gray(cv2.imread(str(p))) for p in frames[::self.ref_stride]]
            self.ref = np.median(np.stack(sample), axis=0).astype(np.float32)
        if self.method == "background" and background is not None:
            self.bg = cv2.GaussianBlur(cv2.imread(str(background), cv2.IMREAD_GRAYSCALE), (5, 5), 0)

    def detect(self, frame_bgr: np.ndarray, prev: np.ndarray | None = None,
               nxt: np.ndarray | None = None) -> Detection | None:
        m = self.method
        if m in ("local_contrast", "background"):
            import analyze  # the existing script; imported lazily for the baseline
            if m == "local_contrast":
                c = analyze.find_led_local_contrast(frame_bgr)
            else:
                c = analyze.find_blob_center(frame_bgr, self.bg)
            if c is None:
                return None
            return Detection(float(c[0]), float(c[1]), 1.0, 3.0, 1.0, 0.0, 0.0)
        lin = linear_gray(frame_bgr)
        if m in ("ref", "ref_dog"):
            a, b = fit_gain(lin, self.ref)
            res = lin - (a * self.ref + b)
        elif m == "diff":
            neigh = [linear_gray(f) for f in (prev, nxt) if f is not None]
            base = np.mean(neigh, axis=0) if neigh else self.ref
            a, b = fit_gain(lin, base)
            res = lin - (a * base + b)
        else:
            raise ValueError(m)
        score = None
        if m == "ref_dog":
            score = cv2.GaussianBlur(res, (0, 0), 1.5) - cv2.GaussianBlur(res, (0, 0), 6.0)
        return locate(res, score, raw=lin)


# Logistic confidence model. Coefficients are fitted on synthetic data by
# `calibrate.py bench detect --fit-confidence` and stored here.
CONF_MODEL_PATH = Path(__file__).with_name("confidence_model.json")


def load_conf_model() -> dict | None:
    if CONF_MODEL_PATH.exists():
        return json.loads(CONF_MODEL_PATH.read_text())
    return None


def confidence(det: Detection, model: dict | None) -> float:
    if model is None:
        return float(1 / (1 + np.exp(-(3 * (det.log_snr - 1.2) + 4 * (det.compact - 0.3) - 4 * det.second))))
    m = model
    z = m["bias"] + sum(m["w"][f] * (getattr(det, f) - m["mu"][f]) / m["sd"][f] for f in FEATURES)
    return float(1 / (1 + np.exp(-z)))


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float = 1e-2, iters: int = 500) -> dict:
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w = np.zeros(Z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = p - y
        H = (Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(len(w))
        w -= np.linalg.solve(H, Z.T @ g + l2 * w)
        b -= g.mean() / max((p * (1 - p)).mean(), 1e-6)
    return {"bias": float(b), "w": dict(zip(FEATURES, map(float, w))),
            "mu": dict(zip(FEATURES, map(float, mu))), "sd": dict(zip(FEATURES, map(float, sd)))}


def frame_change(a_bgr: np.ndarray, b_bgr: np.ndarray, step: int = 2) -> float:
    """Largest local change between two frames after matching exposure."""
    a = linear_gray(a_bgr[::step, ::step])
    b = linear_gray(b_bgr[::step, ::step])
    g = (np.median(a) + 1e-4) / (np.median(b) + 1e-4)
    return float(cv2.GaussianBlur(np.abs(a - g * b), (0, 0), 1.5).max())


def led_signal(frame_bgr: np.ndarray, ref: np.ndarray, x: float, y: float, r: int = 3) -> float:
    """Reference-subtracted brightness around (x, y), for checking an LED is still lit."""
    lin = linear_gray(frame_bgr)
    a, b = fit_gain(lin, ref)
    h, w = lin.shape
    xi, yi = int(round(x)), int(round(y))
    win = (lin - (a * ref + b))[max(yi - r, 0):min(yi + r + 1, h), max(xi - r, 0):min(xi + r + 1, w)]
    return float(win.max()) if win.size else 0.0


def detect_sequence(seq_dir: Path, method: str = "ref_dog", lag: int = 0, n_leds: int = 500,
                    conf_model: dict | None = None, progress: bool = False,
                    track_repeats: bool = True, repeat_ratio: float = 0.2) -> dict:
    """Run a detector over captures/<view>/sequential. Returns a detections dict.

    lag: frame led_XXXX shows LED XXXX - lag at the start of the capture (the
    captures made before the fix have lag 1). With track_repeats, a frame that
    repeats its predecessor (the camera delivered the same image twice) is
    dropped and the lag grows by one from there on.
    """
    frames = sorted(seq_dir.glob("led_*.jpg"))
    det = StillDetector(method)
    det.prepare(frames, seq_dir / "background.jpg")
    images = {}

    def load(i):
        if 0 <= i < len(frames):
            if i not in images:
                images[i] = cv2.imread(str(frames[i]))
            return images[i]
        return None

    out = {}
    shape = None
    repeats = []
    changes = []
    cur_lag = lag
    prev = None  # (detection, its reference-subtracted signal, image) for the last kept frame
    for i, path in enumerate(frames):
        img = load(i)
        shape = img.shape
        d = det.detect(img, load(i - 1), load(i + 1)) if method == "diff" else det.detect(img)
        if track_repeats and det.ref is not None:
            # The previous LED should now be off. If it is still lit at the same
            # brightness, the new peak is in the same place, and the whole image
            # barely changed, the camera delivered the same frame again.
            ch = frame_change(img, load(i - 1)) if i > 0 else np.inf
            if prev is not None and d is not None and len(changes) > 10:
                same_spot = np.hypot(d.x - prev[0].x, d.y - prev[0].y) < 2.0
                still_lit = led_signal(img, det.ref, prev[0].x, prev[0].y) > (1 - repeat_ratio) * prev[1]
                if same_spot and still_lit and ch < 0.6 * np.median(changes[-60:]):
                    repeats.append(i)
                    cur_lag += 1
                    images.pop(i - 2, None)
                    continue
            if np.isfinite(ch):
                changes.append(ch)
            prev = None
            if d is not None and confidence(d, conf_model) >= 0.7:
                prev = (d, led_signal(img, det.ref, d.x, d.y))
        images.pop(i - 2, None)
        led = int(path.stem.split("_")[1]) - cur_lag
        if not 0 <= led < n_leds:
            continue
        if d is None:
            out[led] = None
            continue
        d.conf = 1.0 if method in ("local_contrast", "background") else confidence(d, conf_model)
        out[led] = d
        if progress and i % 100 == 0:
            print(f"  {seq_dir.parent.name}: frame {i}/{len(frames)}")
    for led in range(n_leds):
        out.setdefault(led, None)
    return {"method": method, "lag": lag, "repeated_frames": repeats, "final_lag": cur_lag,
            "image_size": [shape[1], shape[0]], "detections": out}


def save_detections(result: dict, path: Path, view: str, extra: dict | None = None):
    dets = {}
    for led, d in sorted(result["detections"].items()):
        dets[str(led)] = None if d is None else {
            "x": round(d.x, 3), "y": round(d.y, 3), "conf": round(d.conf, 4),
            **{f: round(getattr(d, f), 4) for f in FEATURES}}
    data = {"view": view, "method": result["method"], "lag": result["lag"],
            "repeated_frames": result.get("repeated_frames", []), "final_lag": result.get("final_lag"),
            "image_size": result["image_size"], **(extra or {}), "detections": dets}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1))


def load_detections(path: Path, min_conf: float = 0.0) -> dict:
    data = json.loads(Path(path).read_text())
    uv = np.full((len(data["detections"]), 2), np.nan)
    conf = np.zeros(len(data["detections"]))
    for k, d in data["detections"].items():
        if d is not None and d["conf"] >= min_conf:
            uv[int(k)] = (d["x"], d["y"])
            conf[int(k)] = d["conf"]
    data["uv"], data["conf"] = uv, conf
    return data
