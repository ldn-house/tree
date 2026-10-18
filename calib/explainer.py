"""The flatmate explainer video (docs/calibrate-the-tree.md).

One continuous camera journey around a tree of light: orbit it, push in on one
blinking light, rise overhead to show where to stand, fly into the front
phone's view to show the decoding, pull back as the views cross in 3D, then
play effects on the solved coordinates.

The tree is the synthetic ground-truth layout. The IDs labelled in the phone
view are the ones calib.video decoded from a simulated clip, and the effects
run on the coordinates calib.solve produced from four such clips. Lights are
rendered as additive light with depth of field, multi-scale bloom and a filmic
tone curve.

    ./calibrate.py explainer --clips .scratch/vid/full_h100_b47812 --out docs/img/calibration
"""

import colorsys
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, fields
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H, FPS = 1920, 1080, 30
DURATION = 70.0

# Palette (sRGB)
SPRUCE_CORE = (14, 40, 36)
SPRUCE_EDGE = (5, 15, 14)
CANDLE = (255, 194, 122)
CAL_WHITE = (232, 238, 255)
FROST = (156, 201, 232)
PAPER = (246, 235, 221)
MOSS = (92, 132, 110)

GILL = Path("/System/Library/Fonts/Supplemental/GillSans.ttc")
_FONTS = {}


def font(size: int, weight: str = "semibold") -> ImageFont.FreeTypeFont:
    key = (size, weight)
    if key not in _FONTS:
        if GILL.exists():
            _FONTS[key] = ImageFont.truetype(str(GILL), size, index={"regular": 0, "semibold": 4, "light": 7}[weight])
        else:  # portable fallback
            import matplotlib
            ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
            name = "DejaVuSans-Bold.ttf" if weight == "semibold" else "DejaVuSans.ttf"
            _FONTS[key] = ImageFont.truetype(str(ttf / name), int(size * 0.9))
    return _FONTS[key]


def lin(c):
    c = np.asarray(c, np.float32) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4).astype(np.float32)


def to_srgb8(x):
    x = np.clip(x, 0, 1)
    s = np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)
    return (s * 255 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------- easing

def clamp01(x):
    return float(min(1.0, max(0.0, x)))


def smooth(x):
    x = clamp01(x)
    return x * x * x * (x * (6 * x - 15) + 10)


def span(t, a, b):
    """0 before a, 1 after b, smooth in between."""
    return smooth((t - a) / max(b - a, 1e-6))


def window(t, a, b, fade=0.5):
    return min(span(t, a, a + fade), 1 - span(t, b - fade, b))


# ---------------------------------------------------------------- camera

@dataclass
class Shot:
    yaw: float          # degrees; 0 puts the camera on +x (the front)
    elev: float         # degrees above horizontal
    dist: float         # metres from the target
    ty: float           # target height
    fov: float          # vertical field of view, degrees
    shift: float = 0.12  # lens shift as a fraction of width (tree right of centre)
    tx: float = 0.0
    tz: float = 0.0

    def mix(self, o: "Shot", a: float) -> "Shot":
        return Shot(*(getattr(self, f.name) + (getattr(o, f.name) - getattr(self, f.name)) * a
                      for f in fields(self)))

    def pose(self):
        y, e = np.radians(self.yaw), np.radians(self.elev)
        target = np.array([self.tx, self.ty, self.tz])
        pos = target + self.dist * np.array([np.cos(e) * np.cos(y), np.sin(e), np.cos(e) * np.sin(y)])
        return pos, target


class Cam:
    def __init__(self, s: Shot):
        self.pos, target = s.pose()
        fwd = target - self.pos
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0, 1, 0])
        right /= np.linalg.norm(right)
        down = np.cross(fwd, right)
        self.R = np.stack([right, down, fwd])
        self.f = (H / 2) / np.tan(np.radians(s.fov) / 2)
        self.cx, self.cy = W * (0.5 + s.shift), H / 2
        self.focus = s.dist

    def project(self, P):
        pc = (np.atleast_2d(P) - self.pos) @ self.R.T
        z = pc[:, 2]
        ok = z > 0.05
        zz = np.where(ok, z, 1)
        return self.f * pc[:, 0] / zz + self.cx, self.f * pc[:, 1] / zz + self.cy, np.where(ok, z, np.nan)


# Camera keyframes: (time, shot, easing). "lin" segments keep a steady orbit.
PHONE_SPOT = Shot(yaw=0, elev=11.3, dist=2.04, ty=0.85, fov=60, shift=0.10)
KEYS = [
    (0.0, Shot(-28, 7, 3.7, 0.86, 38), "lin"),
    (14.0, Shot(18, 7, 3.5, 0.86, 38), "smooth"),
    (16.6, Shot(22, 7, 3.4, 0.86, 38), "smooth"),
    (23.2, None, "smooth"),  # push-in: filled in from the chosen light
    (25.0, None, "smooth"),
    (27.6, Shot(36, 12, 4.4, 0.75, 40), "smooth"),
    (31.0, Shot(40, 74, 7.2, 0.35, 40, 0.10), "lin"),
    (37.2, Shot(52, 74, 7.2, 0.35, 40, 0.10), "smooth"),
    (40.4, PHONE_SPOT, "smooth"),
    (46.0, PHONE_SPOT, "smooth"),
    (49.5, Shot(-34, 24, 5.2, 0.8, 40), "lin"),
    (55.5, Shot(-70, 14, 4.0, 0.86, 38), "lin"),
    (64.5, Shot(-112, 8, 3.6, 0.86, 38), "lin"),
    (70.0, Shot(-126, 8, 3.7, 0.86, 38), "lin"),
]


def shot_at(t, keys):
    for (t0, s0, ease), (t1, s1, _) in zip(keys[:-1], keys[1:]):
        if t <= t1:
            a = (t - t0) / max(t1 - t0, 1e-6)
            a = clamp01(a) if ease == "lin" else smooth(a)
            return s0.mix(s1, a)
    return keys[-1][1]


# ---------------------------------------------------------------- light rendering

def _blur(img, sigma):
    if sigma < 5:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = int(min(8, sigma // 2.5))
    small = cv2.resize(img, (W // f, H // f), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / f)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def _splat(u, v, val):
    img = np.zeros((H, W, 3), np.float32)
    ok = np.isfinite(u) & np.isfinite(v) & (u >= 1) & (u < W - 2) & (v >= 1) & (v < H - 2)
    if not ok.any():
        return img
    u, v, val = u[ok], v[ok], val[ok]
    x0, y0 = np.floor(u).astype(int), np.floor(v).astype(int)
    fx, fy = (u - x0)[:, None], (v - y0)[:, None]
    for dx, dy, w in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)), (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        np.add.at(img, (y0 + dy, x0 + dx), val * w)
    return img


def render_lights(cam: Cam, P, C, gain=3.0):
    """Additive point lights with a cheap depth of field (four focus layers).

    In focus, a light's peak equals its value; out of focus, the same energy
    spreads into a bokeh disc, so defocused lights get dimmer and larger.
    """
    u, v, z = cam.project(P)
    ok = np.isfinite(z) & (C.max(1) > 1e-4)
    size = np.where(ok, cam.f / 900 / np.maximum(z, 0.3), 1.0)
    val = C * (gain * np.clip(size, 0.3, 3.0) ** 0.8)[:, None]
    coc = np.where(ok, np.abs(z - cam.focus) / np.maximum(z, 0.3), 0)
    img = np.zeros((H, W, 3), np.float32)
    core = 1.1
    for lo, hi, sigma in ((0, 0.05, core), (0.05, 0.14, 2.4), (0.14, 0.32, 5.0), (0.32, 99, 9.0)):
        m = ok & (coc >= lo) & (coc < hi)
        if m.any():
            s = sigma * float(np.clip(np.median(size[m]), 0.8, 3.0))
            # Normalise to the in-focus peak: energy is conserved as the disc grows.
            boost = (s / core) ** 0.6  # keep bokeh discs visible, not physically dimmed away
            img += _blur(_splat(u[m], v[m], val[m]), s) * (2 * np.pi * core * core) * boost
    return img


def render_glow(cam: Cam, P, C, sigma=40.0):
    """A soft diffuse glow (light on the floor), not individual points."""
    u, v, z = cam.project(P)
    return _blur(_splat(u, v, C.astype(np.float32)), sigma)


def bloom(img):
    return img + 0.6 * _blur(img, 6) + 0.4 * _blur(img, 20) + 0.25 * _blur(img, 60)


def tonemap(x):
    return 1 - np.exp(-x)


def background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r = np.hypot((xx - W * 0.62) / W, (yy - H * 0.6) / H) * 1.5
    t = np.clip(r, 0, 1)[..., None] ** 1.2
    bg = lin(SPRUCE_CORE) * (1 - t) + lin(SPRUCE_EDGE) * t
    return (bg + np.random.default_rng(0).normal(0, 0.0005, (H, W, 1))).astype(np.float32)


def lines_layer(cam: Cam, segs, colour, alpha):
    """Thin additive lines (rays) between 3D point pairs."""
    img = np.zeros((H, W, 3), np.float32)
    if alpha <= 0.01 or not len(segs):
        return img
    a, b = np.array([s[0] for s in segs]), np.array([s[1] for s in segs])
    ua, va, za = cam.project(a)
    ub, vb, zb = cam.project(b)
    col = tuple(float(c) * alpha for c in colour)
    for x0, y0, x1, y1, z0, z1 in zip(ua, va, ub, vb, za, zb):
        if np.isfinite(z0) and np.isfinite(z1):
            cv2.line(img, (int(x0 * 4), int(y0 * 4)), (int(x1 * 4), int(y1 * 4)), col, 1, cv2.LINE_AA, shift=2)
    return img


# ---------------------------------------------------------------- UI

def text(d: ImageDraw.ImageDraw, xy, s, f, fill, alpha):
    if alpha > 0.01:
        d.text(xy, s, font=f, fill=tuple(fill) + (int(255 * alpha),))


def wrap(s, f, width):
    d = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines, cur = [], ""
    for w in s.split():
        t = (cur + " " + w).strip()
        if d.textlength(t, font=f) <= width:
            cur = t
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def caption(layer, s, alpha, x=120, bottom=950, size=64, width=760):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(layer)
    f = font(size)
    lh = int(size * 1.16)
    lines = wrap(s, f, width)
    y = bottom - lh * len(lines)
    for line in lines:
        text(d, (x, y), line, f, PAPER, alpha)
        y += lh


_SCRIM = None


def scrim(layer, strength):
    global _SCRIM
    if strength <= 0.01:
        return
    if _SCRIM is None:
        m = Image.new("L", (W, H), 0)
        ImageDraw.Draw(m).ellipse([-620, 520, 1180, 1560], fill=200)
        _SCRIM = m.filter(ImageFilter.GaussianBlur(140))
    dark = Image.new("RGBA", (W, H), SPRUCE_EDGE + (0,))
    dark.putalpha(_SCRIM.point(lambda v: int(v * strength)))
    layer.alpha_composite(dark)


STEPS = ["Press", "Film", "Read", "Combine"]


def progress(layer, current, alpha):
    if alpha <= 0.01:
        return
    d = ImageDraw.Draw(layer)
    f = font(32, "regular")
    x = 124
    for i, name in enumerate(STEPS):
        on = clamp01(1 - abs(current - i) * 1.6)
        col = tuple(int(MOSS[k] + (CANDLE[k] - MOSS[k]) * on) for k in range(3))
        r = 6 + 3 * on
        d.ellipse([x - r, 90 - r, x + r, 90 + r], fill=col + (int(255 * alpha),))
        text(d, (x + 22, 72), name, f, col, alpha)
        x += 22 + int(d.textlength(name, font=f)) + 56


def phone_glyph(d, x, y, h, alpha, ring=None):
    w = h * 0.52
    a = int(255 * alpha)
    d.rounded_rectangle([x - w / 2, y - h / 2, x + w / 2, y + h / 2], radius=h * 0.14, fill=FROST + (a,))
    d.rounded_rectangle([x - w / 2 + 3, y - h / 2 + 5, x + w / 2 - 3, y + h / 2 - 5], radius=h * 0.09,
                        fill=(16, 34, 44, a))
    if ring is not None and ring > 0:
        rr = h * 0.9
        d.arc([x - rr, y - rr, x + rr, y + rr], -90, -90 + 360 * ring, fill=CANDLE + (a,), width=6)


# ---------------------------------------------------------------- the film

class Film:
    """Everything that does not change from frame to frame."""

    def __init__(self, data: dict):
        self.P = np.array(data["truth"], np.float32)            # world metres, y up
        self.Ps = np.array(data["solved_world"], np.float32)    # solved, aligned to the truth
        self.ys = np.array(data["solved_y"], np.float32)        # contract y in [0, 1]
        self.th = np.array(data["solved_theta"], np.float32)
        self.ids = set(data["decoded_ids"])
        self.rgb = np.array(data["plan_rgb"], np.float32)       # (F, n, 3) 0-255
        self.starts = np.array(data["plan_starts"], np.float32)
        self.dur = np.array(data["plan_dur"], np.float32)
        self.roles = data["plan_roles"]
        self.n = len(self.P)
        rng = np.random.default_rng(7)
        self.phase = rng.uniform(0, 2 * np.pi, self.n)
        self.twf = rng.uniform(0.6, 1.4, self.n)
        self.bg = background()
        self.warm = lin(CANDLE)
        self.white = lin(CAL_WHITE)
        self.code = [i for i, r in enumerate(self.roles) if r == "code"]
        self.sync = [i for i, r in enumerate(self.roles) if r == "sync"]
        self.code_level = max(self.rgb[f].max() for f in self.code)
        # Phones around the tree, front first, 2 m out, at hand height.
        ang = np.radians(np.arange(5) * 72.0)
        self.phones = np.stack([2.0 * np.cos(ang), np.full(5, 1.25), 2.0 * np.sin(ang)], 1)
        # The light we follow: front-ish (seen from the push-in), mid height, near the outside.
        d = np.array([np.cos(np.radians(30)), 0, np.sin(np.radians(30))])
        r = self.P[:, [0, 2]] @ d[[0, 2]]
        cand = np.nonzero((self.P[:, 1] > 0.75) & (self.P[:, 1] < 1.0))[0]
        self.hero = int(cand[np.argmax(r[cand])])
        h = self.P[self.hero]
        self.keys = list(KEYS)
        self.keys[3] = (23.2, Shot(30, 4, 1.15, float(h[1]), 32, 0.10, float(h[0]), float(h[2])), "smooth")
        self.keys[4] = (25.0, Shot(30, 4, 1.15, float(h[1]), 32, 0.10, float(h[0]), float(h[2])), "smooth")
        # Rays and scattering for the "views combine" beat.
        self.spread = rng.uniform(0.55, 1.6, self.n)
        self.settle = rng.uniform(0, 1, self.n)
        self.floor = np.stack([np.sqrt(rng.uniform(0, 1, 260)) * 0.8, np.zeros(260), np.zeros(260)], 1)
        a = rng.uniform(0, 2 * np.pi, 260)
        self.floor = np.stack([self.floor[:, 0] * np.cos(a), np.full(260, 0.01), self.floor[:, 0] * np.sin(a)], 1)
        # Lights decoded in the front clip, spread out for labels.
        self.label_ids = self._pick_labels()

    # -- timing of the calibration sequence on screen --
    BLINK_START = 16.7  # blinking starts after the tap and a short dark pause
    SLOW = 2.0          # the code is shown slowed down while we follow one light

    def _hero_end(self):
        code0 = self.starts[self.code[0]]
        code1 = self.starts[self.code[-1]] + self.dur[self.code[-1]]
        return self.BLINK_START + (code1 - code0) * self.SLOW / 1000

    def seq_frame_at(self, t):
        """Sequence frame shown at film time t (None before it starts).

        The film shows only the code frames: about half the lights are on in
        each, so the tree never flashes as a whole. (The real sequence also has
        whole-tree sync flashes and a dark lead; on screen those would strobe
        and read as dead air.) The first pass is slowed down while we follow
        one light; after that the code cycles at real speed.
        """
        if t < self.BLINK_START:
            return None
        k_slow = int((t - self.BLINK_START) * 1000 / (100 * self.SLOW))
        if k_slow < len(self.code):
            return self.code[k_slow]
        k = int((t - self._hero_end()) * 1000 / 100) % len(self.code)
        return self.code[k]

    def hero_code_shown(self, t):
        """Code frames of the followed light shown so far, during the slowed segment."""
        if t < self.BLINK_START:
            return []
        if t >= self._hero_end():
            return list(self.code)
        f = self.seq_frame_at(t)
        return [g for g in self.code if g <= f]

    def _pick_labels(self, k=9):
        cam = Cam(PHONE_SPOT)
        u, v, z = cam.project(self.P)
        rng = np.random.default_rng(3)
        used, out = set(), []
        for i in rng.permutation(self.n):
            if i in self.ids and np.isfinite(z[i]):
                cell = (int(u[i] // 150), int(v[i] // 120))
                if cell not in used and abs(u[i] - cam.cx) < 230 and 120 < v[i] < 960:
                    used.add(cell)
                    out.append(int(i))
            if len(out) >= k:
                break
        return out

    # -- lights --
    def lights(self, t):
        """Positions (n, 3) and linear colours (n, 3) at film time t."""
        P = self.P
        tw = 0.78 + 0.22 * np.sin(self.phase + t * 2.2 * self.twf)
        warm = np.outer(tw, self.warm) * 1.0
        if t < 7.4:
            return P, warm
        if t < 14.2:
            # A pulse runs up the wire, bottom to top.
            head = (t - 7.6) / 5.6 * self.n
            idx = np.arange(self.n)
            pulse = np.exp(-0.5 * ((idx - head) / 9) ** 2)
            trail = np.where(idx < head, np.exp(-(head - idx) / 45), 0) * 0.8
            base = 0.22 + 0.78 * (1 - window(t, 7.4, 14.2, 0.9))
            c = warm * base + np.outer(pulse * 4.5 + trail, lin((255, 230, 190)))
            return P, c
        if t < 15.9:
            return P, warm
        f = self.seq_frame_at(t)
        if f is None:
            dark = 1 - span(t, 15.9, 16.3)
            return P, warm * dark
        on = self.rgb[f].max(1) / self.code_level
        cal = np.outer(np.clip(on, 0, 1.2), self.white) * 1.15
        if t < 46.0:
            return P, cal
        # Views combine: spread along the front phone's rays, then settle onto the solved positions.
        ph = self.phones[0]
        s_out = span(t, 46.0, 47.8)
        Pspread = ph + (P - ph) * (1 + (self.spread - 1) * s_out)[:, None]
        conv = np.clip((t - 48.6 - self.settle * 3.2) / 1.4, 0, 1)
        conv = conv * conv * (3 - 2 * conv)
        Pt = Pspread + (self.Ps - Pspread) * conv[:, None]
        c = cal * (1 - conv)[:, None] * 0.8 + warm * conv[:, None] * 1.25
        if t < 55.5:
            return Pt, c
        y, th = self.ys, self.th
        if t < 60.4:
            h = 0.5 - 0.58 * np.cos(2 * np.pi * (t - 55.5) / 4.9)
            band = np.exp(-0.5 * ((y - h) / 0.05) ** 2)
            mint = lin((120, 255, 196))
            c = np.outer(band * 4.5, mint) + warm * 0.28
            return self.Ps, np.nan_to_num(c)
        if t < 64.6:
            hue = (th / (2 * np.pi) + y * 1.3 - (t - 60.4) * 0.45) % 1.0
            rgb = np.array([colorsys.hsv_to_rgb(float(hh), 0.85, 1.0) for hh in np.nan_to_num(hue)], np.float32)
            mix = span(t, 60.4, 61.0)
            c = lin(rgb * 255) * 1.1 * mix + warm * 0.05 * (1 - mix)
            return self.Ps, np.nan_to_num(c)
        mix = span(t, 64.6, 65.6)
        hue = (th / (2 * np.pi) + y * 1.3 - (t - 60.4) * 0.45) % 1.0
        rgb = np.array([colorsys.hsv_to_rgb(float(hh), 0.85, 1.0) for hh in np.nan_to_num(hue)], np.float32)
        return self.Ps, np.nan_to_num(lin(rgb * 255) * 1.1 * (1 - mix) + warm * mix)

    def rays(self, t, P):
        segs, alpha = [], 0.0
        if 46.0 <= t < 54.5:
            rng = np.random.default_rng(int(t * 2) % 3)
            phones = [0] if t < 48.6 else range(5)
            for k in phones:
                ph = self.phones[k]
                facing = ((P - ph) * (np.array([0, 0.85, 0]) - ph)).sum(1) > 0
                near = np.nonzero(facing & ((P[:, [0, 2]] * ph[[0, 2]]).sum(1) > 0))[0]
                for i in near[:: max(1, len(near) // (18 if t < 48.6 else 9))]:
                    segs.append((ph, P[i]))
            alpha = window(t, 46.0, 54.5, 0.8) * (0.42 if t < 48.6 else 0.22)
        return segs, alpha

    # -- one frame --
    def frame(self, t) -> np.ndarray:
        s = shot_at(t, self.keys)
        cam = Cam(s)
        P, C = self.lights(t)
        img = render_lights(cam, P, C.astype(np.float32), gain=1.0)
        glow = float(np.nanmean(C))
        img += render_glow(cam, self.floor, np.outer(np.full(len(self.floor), glow * 6.0), self.warm))
        segs, a = self.rays(t, P)
        img += lines_layer(cam, segs, lin(FROST) * 1.4, a)
        img = bloom(img) + self.bg
        rgb = to_srgb8(tonemap(img))
        layer = Image.fromarray(rgb).convert("RGBA")
        self.ui(layer, t, cam)
        return np.asarray(layer.convert("RGB"))

    # -- UI per beat --
    def ui(self, layer, t, cam):
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # E: the phone frame around the front view
        pm = window(t, 39.2, 46.6, 1.0)
        if pm > 0.01:
            sh, sw = 960, 540
            x0, y0 = int(W * 0.6 - sw / 2), int((H - sh) / 2)
            mask = Image.new("L", (W, H), int(205 * pm))
            ImageDraw.Draw(mask).rounded_rectangle([x0, y0, x0 + sw, y0 + sh], radius=48, fill=0)
            dark = Image.new("RGBA", (W, H), SPRUCE_EDGE + (0,))
            dark.putalpha(mask)
            layer.alpha_composite(dark)
            d.rounded_rectangle([x0 - 16, y0 - 16, x0 + sw + 16, y0 + sh + 16], radius=60,
                                outline=FROST + (int(230 * pm),), width=7)
            d.rounded_rectangle([x0 + sw / 2 - 70, y0 - 2, x0 + sw / 2 + 70, y0 + 26], radius=14,
                                fill=SPRUCE_EDGE + (int(255 * pm),))
            # The computer reads the clip: a scan line, then labels pop onto lights.
            scan = span(t, 40.8, 42.4)
            u, v, z = cam.project(self.P)
            ys = y0 + scan * sh
            if 40.8 < t < 42.6:
                d.line([x0, ys, x0 + sw, ys], fill=CANDLE + (int(160 * pm),), width=3)
            for i in self.ids:
                if np.isfinite(z[i]) and v[i] < ys and x0 < u[i] < x0 + sw:
                    d.ellipse([u[i] - 4, v[i] - 4, u[i] + 4, v[i] + 4], fill=CANDLE + (int(150 * pm),))
            f = font(34)
            for k, i in enumerate(self.label_ids):
                a = span(t, 42.6 + 0.28 * k, 42.9 + 0.28 * k) * pm
                if a > 0.01 and np.isfinite(z[i]):
                    lbl = str(i)
                    tw_ = d.textlength(lbl, font=f)
                    bx, by = u[i] + 12, v[i] - 18
                    d.ellipse([u[i] - 11, v[i] - 11, u[i] + 11, v[i] + 11], outline=PAPER + (int(255 * a),), width=3)
                    d.rounded_rectangle([bx, by, bx + tw_ + 18, by + 40], radius=20,
                                        fill=(10, 24, 22, int(225 * a)), outline=CANDLE + (int(255 * a),), width=2)
                    text(d, (bx + 9, by + 1), lbl, f, PAPER, a)
        layer.alpha_composite(ov)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        # C: the Home Assistant button, tapped
        hb = window(t, 14.4, 17.6, 0.5)
        if hb > 0.01:
            x, y, w_, h_ = 120, 560, 640, 128
            d.rounded_rectangle([x, y, x + w_, y + h_], radius=28, fill=(22, 54, 50, int(240 * hb)),
                                outline=MOSS + (int(255 * hb),), width=2)
            tri = [(x + 44, y + 40), (x + 44, y + 88), (x + 86, y + 64)]
            d.polygon(tri, fill=FROST + (int(255 * hb),))
            text(d, (x + 116, y + 36), "Play calibration sequence", font(46), PAPER, hb)
            tap = (t - 15.4) / 0.7
            if 0 < tap < 1:
                r = 20 + 120 * tap
                d.ellipse([x + 64 - r, y + 64 - r, x + 64 + r, y + 64 + r],
                          outline=FROST + (int(220 * (1 - tap) * hb),), width=4)
        # C: the followed light's blinks, and the number they spell
        if 18.0 < t < 25.6:
            a = window(t, 18.0, 25.6, 0.5)
            u, v, z = cam.project(self.P[[self.hero]])
            shown = self.hero_code_shown(t)
            bx, by = 120, 520
            text(d, (bx, by - 64), "One light, slowed down", font(36, "regular"), MOSS, a)
            for k, g in enumerate(self.code):
                cx = bx + k * 26
                lit = self.rgb[g][self.hero].max() > 0
                if g in shown:
                    if lit:
                        d.ellipse([cx, by, cx + 18, by + 18], fill=CAL_WHITE + (int(255 * a),))
                    else:
                        d.ellipse([cx, by, cx + 18, by + 18], outline=MOSS + (int(255 * a),), width=2)
            if t > 23.6:
                aa = span(t, 23.6, 24.1) * a
                text(d, (bx, by + 40), f"That's light {self.hero}", font(48), CANDLE, aa)
            # A ring around the light on screen
            if np.isfinite(z[0]):
                d.ellipse([u[0] - 34, v[0] - 34, u[0] + 34, v[0] + 34], outline=CANDLE + (int(200 * a),), width=3)
        # D: phones appear around the tree, one at a time, each recording
        if 27.4 < t < 39.6:
            a = window(t, 27.4, 39.6, 0.6)
            u, v, z = cam.project(self.phones)
            centre = cam.project(np.array([[0, 0.6, 0]]))
            for k in range(5):
                t0 = 28.4 + 1.85 * k
                if t < t0:
                    continue
                pop = span(t, t0, t0 + 0.35)
                ring = clamp01((t - t0 - 0.3) / 1.4)
                if not np.isfinite(z[k]):
                    continue
                # Field-of-view wedge towards the tree
                dx, dy = centre[0][0] - u[k], centre[1][0] - v[k]
                L = np.hypot(dx, dy) + 1e-6
                nx, ny = -dy / L, dx / L
                tip = (u[k] + dx * 0.95, v[k] + dy * 0.95)
                spread_px = 150
                wedge = [(u[k], v[k]), (tip[0] + nx * spread_px, tip[1] + ny * spread_px),
                         (tip[0] - nx * spread_px, tip[1] - ny * spread_px)]
                active = t < t0 + 1.85 or k == 4
                if active:
                    d.polygon(wedge, fill=FROST + (int(34 * a * pop * (1 - span(t, t0 + 1.6, t0 + 1.85) * (k < 4))),))
                phone_glyph(d, u[k], v[k], 60 * (0.6 + 0.4 * pop), a * pop, ring)
                text(d, (u[k] - 10, v[k] + 44), str(k + 1), font(36), PAPER, a * pop)
        # F: the phones stay in view while their rays cross
        if 46.0 < t < 55.0:
            a = window(t, 46.4, 55.0, 0.6)
            u, v, z = cam.project(self.phones)
            for k in range(5):
                if np.isfinite(z[k]) and (k == 0 or t > 48.4):
                    phone_glyph(d, u[k], v[k], 52, a * (1 if k == 0 else span(t, 48.4, 48.9)))
        # Captions and progress
        for (a0, a1, s) in CAPTIONS:
            al = window(t, a0, a1, 0.5)
            if al > 0.01:
                scrim(layer, al)
                caption(ov, s, al)
        progress(ov, step_at(t), window(t, 14.2, 56.0, 0.8))
        # H: the checklist
        if t > 64.6:
            scrim(layer, span(t, 64.6, 65.2))
            text(d, (120, 520), "Tonight's checklist", font(64), CANDLE, span(t, 64.8, 65.4) * (1 - span(t, 69.4, 70)))
            for k, line in enumerate(CHECKLIST):
                a = span(t, 65.4 + 0.45 * k, 65.9 + 0.45 * k) * (1 - span(t, 69.4, 70))
                text(d, (120, 620 + 74 * k), line, font(50, "regular"), PAPER, a)
        layer.alpha_composite(ov)


CAPTIONS = [
    (0.8, 7.1, "Our tree has 500 lights on one long wire."),
    (7.6, 13.8, "To draw shapes with them, it needs to know where each one is."),
    (14.4, 19.4, "Press Play calibration sequence in Home Assistant."),
    (19.8, 25.4, "Every light blinks its own number, again and again."),
    (26.0, 31.4, "Film it from 4–6 spots, about 2 metres away."),
    (31.8, 37.4, "Hold still for about 12 seconds at each spot."),
    (39.6, 46.2, "The computer reads the blinking and finds each light in every clip."),
    (47.0, 55.2, "Views from different spots cross, pinning each light down in 3D."),
    (56.0, 64.2, "Now effects know where every light is."),
]
CHECKLIST = ["A dark room", "One phone, no zoom", "4–6 spots, front first", "Hold still for 12 seconds"]


def step_at(t):
    for t0, t1, k in ((14.2, 25.6, 0), (25.6, 38.6, 1), (38.6, 46.4, 2), (46.4, 99, 3)):
        if t < t1:
            return k + clamp01((t - (t1 - 0.6)) / 0.6) if t > t1 - 0.6 else float(k)
    return 3.0


# ---------------------------------------------------------------- build

_FILM = None


def _init(data):
    global _FILM
    _FILM = Film(data)


def _render(t):
    return _FILM.frame(t).tobytes()


def prepare(clips: Path, work: Path) -> dict:
    from .detect import load_detections, save_detections
    from .layout import get_layout, load_coords
    from .metrics import align
    from .output import write_coords_txt
    from .sequence import SequenceParams, build
    from .solve import Solver, ViewObs, to_tree_frame
    from .video import decode_video

    work.mkdir(parents=True, exist_ok=True)
    truth = json.loads((clips / "truth.json").read_text())
    names = [v["camera"]["name"] for v in truth["views"]]
    plan_kw = dict(truth["plan"])
    plan_kw.pop("sequential", None)
    plan = build(SequenceParams(**plan_kw))
    dets = []
    for nm in names:
        dp = work / f"{nm}_detections.json"
        if not dp.exists():
            save_detections(decode_video(clips / f"{nm}.mp4", plan), dp, nm)
        dets.append(dp)
    coords_path = work / "coords.txt"
    if not coords_path.exists():
        views = []
        for dp in dets:
            d = load_detections(dp)
            views.append(ViewObs(d["view"], d["image_size"][0], d["image_size"][1], d["uv"], d["conf"]))
        sol = Solver(views).solve()
        write_coords_txt(coords_path, to_tree_frame(sol.points, sol)[0])
    norm = load_coords(coords_path)
    layout = get_layout(truth["layout"])
    world, _ = align(norm, layout.points)
    ok = np.isfinite(world).all(1)
    world[~ok] = layout.points[~ok]
    th = np.arctan2(norm[:, 2], norm[:, 0])
    d0 = load_detections(dets[0], min_conf=0.5)
    return {
        "truth": layout.points.tolist(),
        "solved_world": world.tolist(),
        "solved_y": np.nan_to_num(norm[:, 1]).tolist(),
        "solved_theta": np.nan_to_num(th).tolist(),
        "decoded_ids": [int(i) for i in np.nonzero(np.isfinite(d0["uv"]).all(1))[0]],
        "plan_rgb": plan.rgb().tolist(),
        "plan_starts": plan.starts_ms().tolist(),
        "plan_dur": plan.duration_ms.tolist(),
        "plan_roles": plan.roles,
    }


def stills(data: dict, times: list[float], out: Path):
    out.mkdir(parents=True, exist_ok=True)
    film = Film(data)
    for t in times:
        Image.fromarray(film.frame(t)).save(out / f"t{t:05.1f}.jpg", quality=90)


def build_video(clips: Path, out_dir: Path, work: Path, workers: int = 10, only: list[float] | None = None) -> Path:
    data = prepare(clips, work)
    if only:
        stills(data, only, work / "keyframes")
        return work / "keyframes"
    out_dir.mkdir(parents=True, exist_ok=True)
    mp4 = out_dir / "calibrate-the-tree.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-tune", "film",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    times = np.arange(int(DURATION * FPS)) / FPS
    (work / "stills").mkdir(exist_ok=True)
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(data,)) as ex:
        for k, buf in enumerate(ex.map(_render, times, chunksize=8)):
            proc.stdin.write(buf)
            if k % FPS == 0:
                Image.frombytes("RGB", (W, H), buf).save(work / "stills" / f"s{k // FPS:03d}.jpg", quality=88)
    proc.stdin.close()
    proc.wait()
    # GIF preview: a few seconds from each beat, 10 fps, 480 px wide.
    gif = out_dir / "calibrate-the-tree.gif"
    parts = "+".join(f"between(t,{a},{b})" for a, b in ((9, 11), (20, 22.5), (33, 35), (43.5, 45.5), (50.5, 52.5),
                                                          (61, 63)))
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-vf",
                    f"select='{parts}',setpts=N/10/TB,fps=10,scale=480:-1:flags=lanczos,split[a][b];"
                    "[a]palettegen=max_colors=64:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3",
                    str(gif)], check=True)
    print(f"{mp4}: {DURATION:.0f} s, {mp4.stat().st_size / 1e6:.1f} MB; {gif}: {gif.stat().st_size / 1e6:.1f} MB")
    return mp4
