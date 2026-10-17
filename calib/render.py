"""Synthetic camera simulator for LED calibration.

Renders what a webcam or phone sees when the tree shows a given pattern:
point-source LEDs with directional emission, occlusion by the foliage and
trunk, glow scattered by nearby needles, room fill light, lens halo, colour
crosstalk and white balance, sensor noise and saturation, auto-exposure with
lag and wobble, rolling shutter, and JPEG or video compression.

The scene is linear radiance in arbitrary units where a fully driven LED
channel (255) that is unoccluded and facing the camera puts led_gain into
the pixel before blur.
"""

from dataclasses import dataclass, field

import cv2
import numpy as np

from .camera import Camera
from .layout import Layout, TreeBody

# Camera RGB response to the WS2811 red, green and blue emitters (columns).
# Narrow-band LEDs through a Bayer CFA: green and blue overlap most.
DEFAULT_CROSSTALK = np.array([
    [1.00, 0.10, 0.03],
    [0.12, 1.00, 0.30],
    [0.02, 0.22, 1.00],
])


@dataclass
class SimParams:
    psf_sigma: float = 1.3            # core blur (px at 1080 px image width)
    halo: tuple = ((5.0, 0.10), (24.0, 0.04))  # (sigma px @1080, energy fraction)
    glow_sigma_m: float = 0.06        # needle glow spread in the world
    glow_gain: float = 1.0            # glow energy relative to an unoccluded LED
    wall_gain: float = 0.05           # diffuse light thrown on the wall behind
    room_fill: float = 0.0006         # ambient added per unit of total LED drive
    ambient: float = 0.015            # room radiance with every LED off
    ambient_drift: float = 0.15       # per-frame lognormal sigma of ambient
    fixtures: int = 2                 # bright static light fittings in shot
    led_gain: float = 40.0            # peak radiance of a full LED (before blur)
    emission_floor: float = 0.2       # bullet LED output towards its back
    needle_sigma: float = 0.45        # lognormal sigma of per-view needle shading
    hard_occlusion_p: float = 0.06    # chance a needle clump blocks a view
    core_jitter_m: float = 0.004      # emitting spot vs LED centre (12 mm bullet diffuser)
    partial_shift_px: float = 1.5     # centroid shift when needles cover part of the core
    read_noise: float = 0.003
    shot_noise: float = 0.012
    ae: str = "auto"                  # "auto" or "locked"
    ae_target: float = 0.012          # metered mean (linear, after gain)
    ae_speed: float = 0.4             # fraction of log-error corrected per frame
    ae_wobble: float = 0.12           # per-frame lognormal jitter of the gain
    gain_range: tuple = (0.05, 3.0)
    locked_gain: float = 1.0
    crosstalk: np.ndarray = field(default_factory=lambda: DEFAULT_CROSSTALK.copy())
    wb: tuple = (0.75, 1.0, 1.45)     # warm-room white balance (makes LED white look blue)
    exposure_ms: float = 8.0
    readout_ms: float = 20.0          # rolling-shutter top-to-bottom skew
    jpeg_quality: int = 90
    seed: int = 0


def _scale(cam: Camera) -> float:
    return cam.width / 1080.0


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.3:
        return img
    if sigma < 8:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    # Large kernels: blur at reduced resolution (energy preserving).
    f = int(min(8, max(2, sigma // 3)))
    h, w = img.shape[:2]
    small = cv2.resize(img, (max(1, w // f), max(1, h // f)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def _fractal_noise(rng, h, w, octaves=5, base=8):
    """Zero-mean value noise; the coarsest octave has `base` cells on the long side."""
    out = np.zeros((h, w), np.float32)
    amp, total = 1.0, 0.0
    long_side = max(h, w)
    for o in range(octaves):
        cells = base << o
        gh = max(2, int(round(cells * h / long_side)))
        gw = max(2, int(round(cells * w / long_side)))
        g = rng.normal(0, 1, (gh, gw)).astype(np.float32)
        out += amp * cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        total += amp
        amp *= 0.6
    return out / total


def _splat(shape, uv, values):
    """Bilinear splat of (n, C) values at sub-pixel positions into (H, W, C)."""
    h, w = shape
    img = np.zeros((h, w, values.shape[1]), np.float32)
    ok = np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < w - 1) & (uv[:, 1] >= 0) & (uv[:, 1] < h - 1)
    u, v, val = uv[ok, 0], uv[ok, 1], values[ok]
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    for dx, dy, wgt in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                        (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(img, (y0 + dy, x0 + dx), val * wgt)
    return img


def transmission(points: np.ndarray, cam_pos: np.ndarray, body: TreeBody, samples: int = 48) -> np.ndarray:
    """Beer-Lambert transmission from each LED to the camera through foliage and trunk."""
    s = np.linspace(0.02, 1.0, samples)[None, :, None]
    seg = points[:, None, :] + s * (cam_pos[None, None, :] - points[:, None, :])
    r = np.hypot(seg[..., 0], seg[..., 2])
    y = seg[..., 1]
    inside = (r < body.radius(y)) & (y > body.bottom) & (y < body.height)
    seg_len = np.linalg.norm(cam_pos - points, axis=1)
    depth = inside.mean(1) * seg_len
    trunk = ((r < body.trunk_radius) & (y < body.height * 0.92)).any(1)
    return np.exp(-body.density * depth) * np.where(trunk, 0.0, 1.0)


def _cone_mask(cam: Camera, body: TreeBody, scale: int = 4) -> np.ndarray:
    """Foliage silhouette by ray-casting the cone at reduced resolution."""
    h, w = cam.height // scale, cam.width // scale
    uu, vv = np.meshgrid((np.arange(w) + 0.5) * scale, (np.arange(h) + 0.5) * scale)
    d = cam.rays(np.stack([uu.ravel(), vv.ravel()], 1))
    C = cam.centre
    k = body.base_radius / (body.height - body.bottom)
    ya = body.height
    a = d[:, 0] ** 2 + d[:, 2] ** 2 - k * k * d[:, 1] ** 2
    b = 2 * (C[0] * d[:, 0] + C[2] * d[:, 2] + k * k * (ya - C[1]) * d[:, 1])
    c = C[0] ** 2 + C[2] ** 2 - k * k * (ya - C[1]) ** 2
    disc = b * b - 4 * a * c
    hit = np.zeros(len(d), bool)
    ok = disc >= 0
    sq = np.sqrt(np.where(ok, disc, 0))
    for sgn in (-1, 1):
        s = (-b + sgn * sq) / (2 * a)
        y = C[1] + s * d[:, 1]
        hit |= ok & (s > 0) & (y > body.bottom) & (y < ya)
    m = hit.reshape(h, w).astype(np.float32)
    return cv2.resize(m, (cam.width, cam.height), interpolation=cv2.INTER_LINEAR)


@dataclass
class View:
    """Everything about one camera that does not change from frame to frame."""
    cam: Camera
    uv: np.ndarray            # (n, 2) true LED image positions
    emit_uv: np.ndarray       # (n, 2) where each LED's light appears in this view
    depth: np.ndarray         # (n,)
    trans: np.ndarray         # (n,) direct-light transmission incl. needles and emission
    glow_vis: np.ndarray      # (n,)
    background: np.ndarray    # (H, W, 3) static radiance at unit ambient
    fixtures: np.ndarray      # (H, W, 3) light fittings (not scaled by ambient)
    foliage: np.ndarray       # (H, W) mask * texture for glow
    wall: np.ndarray          # (H, W) where wall light lands
    glow_sigma_px: float

    @property
    def visible(self) -> np.ndarray:
        """LEDs a careful human would mark as visible in this view."""
        h, w = self.cam.height, self.cam.width
        inb = np.isfinite(self.uv).all(1) & (self.uv[:, 0] > 2) & (self.uv[:, 0] < w - 3) \
            & (self.uv[:, 1] > 2) & (self.uv[:, 1] < h - 3)
        return inb & (self.trans > 0.08)


def make_view(layout: Layout, cam: Camera, p: SimParams, rng: np.random.Generator) -> View:
    pts = layout.points
    uv, depth = cam.project(pts)
    tr = transmission(pts, cam.centre, layout.body)
    # The bright spot is where the diffuser faces the camera, not the LED centre,
    # and partial cover by needles pulls the centroid sideways. View.uv keeps
    # the true centre; emit_uv is where the light appears.
    emit_uv, _ = cam.project(pts + rng.normal(0, p.core_jitter_m, pts.shape))
    # Bullet LEDs point roughly outwards from the trunk with random tilt.
    out = np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 2]], 1)
    out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-6)
    axis = out + rng.normal(0, 0.7, out.shape)
    axis /= np.linalg.norm(axis, axis=1, keepdims=True)
    to_cam = cam.centre - pts
    to_cam /= np.linalg.norm(to_cam, axis=1, keepdims=True)
    cosang = (axis * to_cam).sum(1)
    emis = p.emission_floor + (1 - p.emission_floor) * np.clip(cosang, 0, 1) ** 0.8
    needles = np.exp(rng.normal(0, p.needle_sigma, len(pts)))
    needles = np.minimum(needles, 1.6) * np.where(rng.random(len(pts)) < p.hard_occlusion_p, 0.03, 1.0)
    trans = tr * emis * needles
    partial = np.clip(1 - needles, 0, 1)
    emit_uv = emit_uv + rng.normal(0, 1, emit_uv.shape) * (p.partial_shift_px * _scale(cam) * (0.3 + partial))[:, None]
    glow_vis = np.clip(tr, 0, 1) ** 0.3 * (0.4 + 0.6 * np.sqrt(emis))

    h, w = cam.height, cam.width
    sc = _scale(cam)
    yy = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    xx = np.linspace(0, 1, w, dtype=np.float32)[None, :]
    wall_shade = 0.6 + 0.4 * np.exp(-((xx - rng.uniform(0.2, 0.8)) ** 2) / 0.15) * (1.1 - 0.4 * yy)
    horizon = rng.uniform(0.78, 0.9)
    floor = yy > horizon
    tex = _fractal_noise(rng, h, w)
    room = np.where(floor, 0.35 + 0.1 * tex, wall_shade * (1 + 0.05 * tex)).astype(np.float32)
    wall_col = np.array(rng.choice([[0.9, 1.0, 1.05], [1.0, 0.95, 0.85], [0.85, 0.95, 0.95]]), np.float32)
    bg = room[..., None] * wall_col
    mask = _cone_mask(cam, layout.body)
    # Ragged branch tiers: medium-scale noise plus horizontal banding near the edge.
    tiers = 0.12 * np.sin(yy * rng.uniform(40, 60) + 2 * _fractal_noise(rng, h, w, octaves=2, base=6))
    ragged = cv2.GaussianBlur(mask, (0, 0), 10 * sc) + 0.35 * _fractal_noise(rng, h, w, octaves=4, base=16) + tiers
    mask = np.clip((ragged - 0.45) * 6, 0, 1).astype(np.float32)
    clumps = _fractal_noise(rng, h, w, octaves=3, base=24)
    needles_hf = _fractal_noise(rng, h, w, octaves=2, base=int(260 * h / 1080))
    needle_tex = np.clip(0.5 + 0.5 * clumps + 0.7 * needles_hf, 0.0, 2.0).astype(np.float32)
    foliage_col = np.array([0.05, 0.09, 0.06], np.float32)
    bg = bg * (1 - mask[..., None]) + mask[..., None] * foliage_col * needle_tex[..., None]
    fixtures = np.zeros_like(bg)
    for _ in range(p.fixtures):
        # Place fittings away from the tree silhouette.
        for _try in range(20):
            fx, fy = rng.uniform(0.05, 0.95) * w, rng.uniform(0.03, 0.6) * h
            if mask[int(fy), int(fx)] < 0.1:
                break
        r = rng.uniform(3, 7) * sc
        dot = np.zeros((h, w), np.float32)
        cv2.circle(dot, (int(fx), int(fy)), int(max(r, 1)), 1.0, -1, cv2.LINE_AA)
        fixtures += _blur(dot, 1.5 * sc)[..., None] * np.array([1.0, 0.85, 0.6], np.float32) * rng.uniform(3, 8)
    glow_sigma = float(np.nanmedian(cam.fx * p.glow_sigma_m / depth))
    return View(cam, uv, emit_uv, depth, trans.astype(np.float32), glow_vis.astype(np.float32), bg.astype(np.float32),
                fixtures.astype(np.float32), (mask * needle_tex).astype(np.float32), 1 - mask, glow_sigma)


class Sensor:
    """Auto-exposure, white balance, noise and tone mapping, with state across frames."""

    def __init__(self, p: SimParams, rng: np.random.Generator):
        self.p = p
        self.rng = rng
        self.gain = p.locked_gain if p.ae == "locked" else None

    def meter(self, scene: np.ndarray) -> float:
        h, w = scene.shape[:2]
        c = scene[h // 6: 5 * h // 6: 4, w // 6: 5 * w // 6: 4].mean()
        return float(0.6 * c + 0.4 * scene[::8, ::8].mean())

    def expose(self, scene: np.ndarray, settle: int = 1) -> np.ndarray:
        p = self.p
        if p.ae == "auto":
            m = max(self.meter(scene), 1e-6)
            if self.gain is None:
                self.gain = float(np.clip(p.ae_target / m, *p.gain_range))
            for _ in range(settle):
                err = np.log(p.ae_target / max(m * self.gain, 1e-9))
                self.gain *= float(np.exp(p.ae_speed * err + self.rng.normal(0, p.ae_wobble)))
                self.gain = float(np.clip(self.gain, *p.gain_range))
        g = self.gain
        raw = scene * g
        raw = raw + self.rng.normal(0, 1, raw.shape).astype(np.float32) * np.sqrt(
            np.maximum(raw, 0) * p.shot_noise ** 2 + p.read_noise ** 2)
        # Sensor saturation and charge bleed into neighbours.
        excess = np.maximum(raw - 1.0, 0)
        if excess.any():
            raw = np.minimum(raw, 1.0) + 0.5 * _blur(excess, 2.0)
        raw = raw * np.array(p.wb, np.float32)
        out = np.clip(raw, 0, 1)
        out = np.where(out < 0.0031308, 12.92 * out, 1.055 * np.power(out, 1 / 2.4) - 0.055)
        return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def radiance(view: View, led_rgb: np.ndarray, p: SimParams, ambient: float,
             band_drives: np.ndarray | None = None) -> np.ndarray:
    """Linear scene radiance (H, W, 3) for per-LED drive levels led_rgb (n, 3), 0-255.

    With a rolling shutter, each image row samples the scene at its own time.
    led_rgb is each LED's drive at its own row, which is right for light that
    lands near the LED (core, halo, needle glow). band_drives (B, n, 3) gives
    every LED's drive at the times of B horizontal bands, for the light that
    spreads far (wall glow, room fill), which is composited band by band.
    """
    cam = view.cam
    sc = _scale(cam)
    shape = (cam.height, cam.width)
    drive = (led_rgb / 255.0) @ p.crosstalk.T  # camera RGB per LED
    direct = _splat(shape, view.emit_uv, drive * (view.trans * p.led_gain)[:, None])
    core = 1 - sum(e for _, e in p.halo)
    img = core * _blur(direct, p.psf_sigma * sc)
    for sigma, e in p.halo:
        img += e * _blur(direct, sigma * sc)
    glow_src = _splat(shape, view.emit_uv, drive * (view.glow_vis * p.led_gain * p.glow_gain)[:, None])
    glow = _blur(glow_src, view.glow_sigma_px) * (view.foliage ** 2)[..., None] * 1.2
    def far_light(dr):
        wall_src = _splat(shape, view.uv, dr * p.led_gain)
        wall = _blur(wall_src, 0.12 * cam.width) * view.wall[..., None] * p.wall_gain * 60
        fill = view.background * (ambient + p.room_fill * dr.sum(0)[None, None, :] / 3)
        return wall + fill

    if band_drives is None:
        far = far_light(drive)
    else:
        B = len(band_drives)
        far = np.empty_like(img)
        edges = np.linspace(0, cam.height, B + 1).astype(int)
        for b, bd in enumerate(band_drives):
            far[edges[b]:edges[b + 1]] = far_light((bd / 255.0) @ p.crosstalk.T)[edges[b]:edges[b + 1]]
    return far + view.fixtures + glow + img


def band_drive(view: View, plan_rgb: np.ndarray, starts_ms: np.ndarray, durations_ms: np.ndarray,
               frame_t_ms: float, p: SimParams, bands: int = 8, loop_ms: float | None = None) -> np.ndarray:
    """Every LED's exposure-integrated drive at the time of each of `bands` row bands."""
    out = []
    for b in range(bands):
        frac = (b + 0.5) / bands
        t = frame_t_ms + frac * p.readout_ms
        out.append(_window_drive(plan_rgb, starts_ms, durations_ms, np.array([t]), p.exposure_ms,
                                 loop_ms)[0])
    return np.stack(out)


def _window_drive(plan_rgb, starts_ms, durations_ms, t0, exposure_ms, loop_ms):
    """Mean drive over [t0, t0 + exposure] for each t0 (shape (k,)), all LEDs: (k, n, 3)."""
    ends = starts_ms + durations_ms
    total = ends[-1]
    t1 = t0 + exposure_ms
    acc = np.zeros((len(t0),) + plan_rgb.shape[1:], np.float32)
    for shift in ((0.0,) if loop_ms is None else (-total, 0.0, total)):
        ov = np.clip(np.minimum(t1[:, None], ends[None] + shift) - np.maximum(t0[:, None], starts_ms[None] + shift),
                     0, None)  # (k, F)
        for f in np.nonzero(ov.any(0))[0]:
            acc += (ov[:, f] / exposure_ms)[:, None, None] * plan_rgb[f][None]
    return acc


def encode_jpeg(img_rgb: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def ambient_series(p: SimParams, n: int, rng: np.random.Generator) -> np.ndarray:
    """Ambient level per frame: slow random walk plus jitter."""
    walk = np.cumsum(rng.normal(0, p.ambient_drift / 4, n))
    walk -= walk.mean()
    return p.ambient * np.exp(walk + rng.normal(0, p.ambient_drift / 4, n))


def rolling_shutter_drive(view: View, plan_rgb: np.ndarray, starts_ms: np.ndarray, durations_ms: np.ndarray,
                          frame_t_ms: float, p: SimParams, loop_ms: float | None = None) -> np.ndarray:
    """Per-LED drive integrated over each LED's exposure window in one video frame.

    The tree switches frames at starts_ms (latch times). An LED at image row v is
    exposed during [t + v/H * readout, + exposure]. Returns (n, 3).
    """
    rows = np.nan_to_num(view.uv[:, 1], nan=view.cam.height / 2) / view.cam.height
    t0 = frame_t_ms + rows * p.readout_ms
    t1 = t0 + p.exposure_ms
    ends = starts_ms + durations_ms
    total = ends[-1]
    acc = np.zeros((len(rows), 3), np.float32)
    for shift in ((0.0,) if loop_ms is None else (-total, 0.0, total)):
        s = starts_ms + shift
        e = ends + shift
        # Overlap of each LED window with each tree frame: (n, F)
        ov = np.clip(np.minimum(t1[:, None], e[None]) - np.maximum(t0[:, None], s[None]), 0, None)
        if not ov.any():
            continue
        f_idx = np.nonzero(ov.any(0))[0]
        for f in f_idx:
            acc += (ov[:, f] / p.exposure_ms)[:, None] * plan_rgb[f]
    return acc
