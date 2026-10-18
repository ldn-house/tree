"""The flatmate explainer video (docs/calibrate-the-tree.md).

Everything shown is synthetic: the simulator's tree, a simulated phone clip
decoded by calib.video, and coordinates solved from four such clips. Frames
are drawn with Pillow and encoded with ffmpeg.

    ./calibrate.py explainer --clips .scratch/vid/full_h100_b47812 --out docs/img/calibration
"""

import json
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .detect import load_detections
from .layout import get_layout, load_coords
from .render import Sensor, SimParams, make_view, radiance
from .sequence import SequenceParams, build
from .synth import make_rig

W, H, FPS = 1920, 1080, 30
BG = (14, 18, 24)
FG = (240, 240, 235)
ACCENT = (255, 196, 92)
DIM = (150, 160, 170)
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"


def font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")), size)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: int) -> list[str]:
    lines = []
    for para in text.split("\n"):
        words, cur = para.split(), ""
        for w in words:
            t = (cur + " " + w).strip()
            if draw.textlength(t, font=f) <= width:
                cur = t
            else:
                lines.append(cur)
                cur = w
        lines.append(cur)
    return lines


def caption(img: Image.Image, title: str, body: str = "", x: int = 760, y: int = 300, width: int = 1060,
            alpha: float = 1.0):
    d = ImageDraw.Draw(img)
    col = lambda c: tuple(int(BG[i] + (c[i] - BG[i]) * alpha) for i in range(3))
    ft, fb = font(66), font(44, bold=False)
    for line in wrap(d, title, ft, width):
        d.text((x, y), line, font=ft, fill=col(ACCENT))
        y += 84
    y += 26
    for line in wrap(d, body, fb, width):
        d.text((x, y), line, font=fb, fill=col(FG))
        y += 60
    return y


def canvas() -> Image.Image:
    return Image.new("RGB", (W, H), BG)


def fade(i: int, n: int, edge: int = 12) -> float:
    return float(min(1.0, (i + 1) / edge, (n - i) / edge))


class TreeRenderer:
    """The simulator's tree from one phone viewpoint, as a 607 x 1080 picture."""

    def __init__(self, seed: int = 1):
        self.layout = get_layout("legacy-cleaned")
        cam = make_rig("phone", self.layout, 4, seed=seed, scale=0.5625)[0]
        self.p = SimParams(seed=seed, ae="locked", locked_gain=0.55, fixtures=0, ambient=0.01)
        self.view = make_view(self.layout, cam, self.p, np.random.default_rng(seed))
        self.sensor = Sensor(self.p, np.random.default_rng(seed + 1))
        self.cam = cam

    def frame(self, rgb: np.ndarray) -> Image.Image:
        img = self.sensor.expose(radiance(self.view, rgb.astype(np.float32), self.p, self.p.ambient))
        return Image.fromarray(img)


def paste_tree(img: Image.Image, tree: Image.Image, x: int = 70):
    img.paste(tree, (x, (H - tree.height) // 2))


def bit_strip(img: Image.Image, bits: list[int], x: int, y: int, label: str, size: int = 44):
    d = ImageDraw.Draw(img)
    d.text((x, y - 66), label, font=font(40), fill=FG)
    for k, b in enumerate(bits):
        x0 = x + k * (size + 10)
        d.rounded_rectangle([x0, y, x0 + size, y + size], radius=8,
                            fill=ACCENT if b else (40, 46, 56), outline=(90, 96, 110), width=2)


def person(d: ImageDraw.ImageDraw, x: float, y: float, facing: float, recording: bool):
    d.ellipse([x - 22, y - 22, x + 22, y + 22], fill=(120, 170, 255))
    px, py = x + 38 * np.cos(facing), y + 38 * np.sin(facing)
    d.rectangle([px - 9, py - 9, px + 9, py + 9], fill=(230, 230, 230))
    if recording:
        d.ellipse([x + 26, y - 46, x + 46, y - 26], fill=(235, 60, 60))


def scene_rest(tree: TreeRenderer, n: int):
    rng = np.random.default_rng(3)
    phase = rng.uniform(0, 2 * np.pi, tree.layout.n)
    for i in range(n):
        tw = 0.7 + 0.3 * np.sin(phase + i / FPS * 3.0)
        rgb = np.outer(tw, [255, 150, 60]) * 0.6
        img = canvas()
        paste_tree(img, tree.frame(rgb))
        caption(img, "How the tree learns where its lights are",
                "500 lights on one long wire. To draw shapes in 3D, the tree needs to know where each one is.",
                alpha=fade(i, n))
        yield img


def scene_code(tree: TreeRenderer, plan, led: int, n_dark: int, slow: float):
    rgb = plan.rgb()
    starts = plan.starts_ms()
    sync = plan.index("sync")
    code = [f for f in range(plan.n_frames) if plan.roles[f] == "code"]
    # Dark wait (shortened), the sync flashes at real speed, then the code slowed down.
    for i in range(n_dark):
        img = canvas()
        paste_tree(img, tree.frame(np.zeros_like(rgb[0])))
        caption(img, "1. Press “Play calibration sequence”",
                "The tree goes dark for a few seconds, then starts blinking.", alpha=fade(i, n_dark + 1000))
        yield img
    t_sync = starts[sync[0]]
    t_end = starts[sync[-1]] + plan.duration_ms[sync[-1]]
    for t in np.arange(t_sync, t_end, 1000 / FPS):
        f = int(np.searchsorted(starts, t, side="right") - 1)
        img = canvas()
        paste_tree(img, tree.frame(rgb[f]))
        caption(img, "1. Press “Play calibration sequence”",
                "A burst of flashes marks the start…")
        yield img
    t0, t1 = starts[code[0]], starts[code[-1]] + plan.duration_ms[code[-1]]
    uv = tree.view.uv[led]
    for t in np.arange(t0, t1, 1000 / FPS / slow):
        f = int(np.searchsorted(starts, t, side="right") - 1)
        img = canvas()
        paste_tree(img, tree.frame(rgb[f]))
        d = ImageDraw.Draw(img)
        cx, cy = 70 + uv[0], (H - tree.cam.height) // 2 + uv[1]
        d.ellipse([cx - 30, cy - 30, cx + 30, cy + 30], outline=ACCENT, width=5)
        caption(img, "Every light blinks its own number",
                f"Slowed down {slow:g}×. Each light flashes a code: on or off, then the opposite, "
                "with extra checks so a misread light is rejected.", y=230)
        shown = [int(rgb[g][led].max() > 0) for g in code if starts[g] <= t]
        bit_strip(img, shown[-16:], 760, 800, f"Light {led} so far:")
        yield img


def scene_filming(n: int, spots: int = 5):
    cx, cy, R = 370, 540, 260
    for i in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        # Tree from above: a soft green disc with a star.
        for r, c in ((140, (24, 70, 40)), (100, (30, 95, 52)), (62, (40, 120, 64))):
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=c)
        d.text((cx - 18, cy - 30), "★", font=font(48), fill=ACCENT)
        t = i / n * spots
        k = min(int(t), spots - 1)
        frac = t - k
        angles = [np.pi / 2 + 2 * np.pi * j / spots for j in range(spots)]
        for j, a in enumerate(angles):
            sx, sy = cx + R * np.cos(a), cy + R * np.sin(a)
            done = j < k or (j == k and frac > 0.75)
            d.ellipse([sx - 34, sy - 34, sx + 34, sy + 34], outline=(90, 96, 110), width=3,
                      fill=(30, 60, 40) if done else None)
            d.text((sx - 11, sy - 21), str(j + 1), font=font(34), fill=FG if not done else (150, 230, 160))
        fx, fy = cx + R * np.cos(angles[0]), cy + R * np.sin(angles[0])
        d.text((fx + 48, fy - 18), "front", font=font(30, bold=False), fill=DIM)
        # Walk for the first 25% of each step, then stand on the spot and record.
        a_prev = angles[k - 1] if k > 0 else angles[0]
        a_now = angles[k]
        if frac < 0.25 and k > 0:
            a = a_prev + (a_now - a_prev) * (frac / 0.25)
            rec = False
        else:
            a, rec = a_now, True
        px, py = cx + R * np.cos(a), cy + R * np.sin(a)
        person(d, px, py, a + np.pi, rec)
        if rec:
            secs = 12 * (frac - 0.25) / 0.75
            tx, ty = px, py + 62  # under the person
            label = f"REC {secs:4.1f} s"
            tw = d.textlength(label, font=font(30))
            tx = min(max(tx - tw / 2, 20), 700 - tw)  # keep it inside the left panel
            d.text((tx, ty - 18), label, font=font(30), fill=(235, 90, 90))
        caption(img, "2. Film it from 4–6 spots",
                "About 2 metres away, front first, the whole tree in the frame. "
                "Hold still for about 12 seconds at each spot, then move to the next.",
                alpha=fade(i, n))
        yield img


def scene_decode(clip: Path, det_path: Path, n: int, labels: int = 14):
    import cv2
    cap = cv2.VideoCapture(str(clip))
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) * 0.55)):
        cap.read()
    ok, fr = cap.read()
    cap.release()
    base = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).resize((607, 1080), Image.LANCZOS)
    base = Image.eval(base, lambda v: int(v * 0.55))
    sc = 1080 / fr.shape[0]
    det = load_detections(det_path, min_conf=0.5)
    pts = [(i, det["uv"][i] * sc) for i in range(len(det["uv"])) if np.isfinite(det["uv"][i]).all()]
    rng = np.random.default_rng(2)
    # Labels spread out: pick from a coarse grid so they don't overlap.
    chosen, used = [], set()
    for i, p in sorted(pts, key=lambda q: rng.random()):
        cell = (int(p[0] // 120), int(p[1] // 110))
        if cell not in used:
            used.add(cell)
            chosen.append((i, p))
        if len(chosen) >= labels:
            break
    hue = lambda i: tuple(int(c * 255) for c in matplotlib.colormaps["turbo"](0.1 + 0.8 * i / 500)[:3])
    for f in range(n):
        img = canvas()
        tile = base.copy()
        d = ImageDraw.Draw(tile)
        shown = int(len(pts) * min(1.0, f / (n * 0.35)))
        for i, p in pts[:shown]:
            d.ellipse([p[0] - 6, p[1] - 6, p[0] + 6, p[1] + 6], fill=hue(i))
        for k, (i, p) in enumerate(chosen):
            if f > n * 0.35 + k * 4:
                d.rounded_rectangle([p[0] + 10, p[1] - 22, p[0] + 92, p[1] + 14], radius=7, fill=(0, 0, 0))
                d.text((p[0] + 16, p[1] - 22), str(i), font=font(30), fill=FG)
                d.ellipse([p[0] - 10, p[1] - 10, p[0] + 10, p[1] + 10], outline=FG, width=3)
        paste_tree(img, tile)
        caption(img, "3. The computer reads the blinking",
                f"In each clip it works out which light is where on the screen: "
                f"{len(pts)} lights found in this one.", alpha=fade(f, n))
        yield img


def _rot(points, yaw, pitch=0.25):
    cy_, sy_ = np.cos(yaw), np.sin(yaw)
    x = points[:, 0] * cy_ + points[:, 2] * sy_
    z = -points[:, 0] * sy_ + points[:, 2] * cy_
    y = points[:, 1]
    cp, sp = np.cos(pitch), np.sin(pitch)
    return x, y * cp - z * sp, y * sp + z * cp


def scene_cloud(coords: np.ndarray, cams: np.ndarray, n: int):
    ok = np.isfinite(coords).all(1)
    p = coords.copy()
    p[:, 1] -= 0.5
    idx = np.arange(len(p))
    cm = matplotlib.colormaps["turbo"]
    for f in range(n):
        yaw = 2 * np.pi * f / n * 0.9
        img = canvas()
        layer = Image.new("RGB", (720, 1000), BG)
        d = ImageDraw.Draw(layer)
        x, y, z = _rot(p[ok], yaw)
        order = np.argsort(z)
        s = 600
        for j in order:
            u, v = 360 + x[j] * s, 500 - y[j] * s
            c = tuple(int(255 * q) for q in cm(0.1 + 0.8 * idx[ok][j] / len(p))[:3])
            r = 5 + 2 * (z[j] + 0.5)
            d.ellipse([u - r, v - r, u + r, v + r], fill=c)
        # Phone spots from the solve, drawn as white markers around the tree.
        cpos = cams.copy()
        cpos[:, 1] -= 0.5
        cx, cyy, cz = _rot(cpos / np.maximum(np.linalg.norm(cpos[:, [0, 2]], axis=1, keepdims=True), 1e-6) * 0.55,
                           yaw)
        for u0, v0 in zip(360 + cx * s, 500 - cyy * s):
            # A small phone: light body, dark screen, at each filming spot.
            d.rounded_rectangle([u0 - 13, v0 - 22, u0 + 13, v0 + 22], radius=6, fill=(120, 170, 255))
            d.rounded_rectangle([u0 - 9, v0 - 16, u0 + 9, v0 + 14], radius=3, fill=(20, 30, 50))
        glow = layer.filter(ImageFilter.GaussianBlur(6))
        layer = Image.blend(layer, glow, 0.35)
        img.paste(layer, (0, 40))
        caption(img, "4. The views combine into 3D",
                "One clip only gives a direction to each light. Clips from several spots cross at each "
                "light and pin it down in space.", alpha=fade(f, n))
        yield img


def scene_sweep(tree: TreeRenderer, coords: np.ndarray, n: int):
    y = coords[:, 1]
    for f in range(n):
        h = 0.5 - 0.55 * np.cos(2 * np.pi * f / (n / 1.5))
        on = np.isfinite(y) & (np.abs(y - h) < 0.09)
        rgb = np.zeros((len(y), 3))
        rgb[on] = [60, 255, 120]
        img = canvas()
        paste_tree(img, tree.frame(rgb * 1.6))
        caption(img, "Done!",
                "Effects now know where every light is. Here a flat plane sweeps up and down the tree, "
                "using the coordinates the computer just worked out.", alpha=fade(f, n))
        yield img


def scene_end(n: int):
    for f in range(n):
        img = canvas()
        d = ImageDraw.Draw(img)
        lines = ["Dark room", "One phone, no zoom", "4–6 spots, front first", "Hold still ~12 s at each"]
        y = 250
        d.text((160, 140), "Recap", font=font(72), fill=ACCENT)
        for k, line in enumerate(lines):
            d.text((160, y + 110 * k), "•  " + line, font=font(54, bold=False), fill=FG)
        d.text((160, 800), "Full guide: docs/calibrate-the-tree.md", font=font(40, bold=False), fill=DIM)
        yield img


def build_video(clips: Path, out_dir: Path, work: Path, seconds_per_scene: dict | None = None) -> Path:
    from .solve import Solver, ViewObs, to_tree_frame
    from .video import decode_video
    from .detect import save_detections

    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = json.loads((clips / "truth.json").read_text())
    names = [v["camera"]["name"] for v in truth["views"]]
    plan_kw = dict(truth["plan"])
    plan_kw.pop("sequential", None)
    plan = build(SequenceParams(**plan_kw))
    det_paths = []
    for nm in names:
        dp = work / f"{nm}_detections.json"
        if not dp.exists():
            res = decode_video(clips / f"{nm}.mp4", plan)
            save_detections(res, dp, nm)
        det_paths.append(dp)
    coords_path = work / "coords.txt"
    cams_path = work / "cams.json"
    if not coords_path.exists():
        views = []
        for dp in det_paths:
            d = load_detections(dp)
            views.append(ViewObs(d["view"], d["image_size"][0], d["image_size"][1], d["uv"], d["conf"]))
        sol = Solver(views).solve()
        norm, info = to_tree_frame(sol.points, sol)
        from .output import write_coords_txt
        write_coords_txt(coords_path, norm)
        cams_path.write_text(json.dumps(info["cameras"]))
    coords = load_coords(coords_path)
    cams = np.array(json.loads(cams_path.read_text()))

    tree = TreeRenderer()
    vis = np.nonzero(tree.view.visible & (tree.view.trans > 0.4))[0]
    led = int(vis[len(vis) // 2])
    mp4 = out_dir / "calibrate-the-tree.mp4"
    proc = subprocess.Popen(
        ["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "26", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", str(mp4)], stdin=subprocess.PIPE)
    gif_frames = []
    scenes = [
        scene_rest(tree, 7 * FPS),
        scene_code(tree, plan, led, n_dark=int(2.5 * FPS), slow=3),
        scene_filming(13 * FPS),
        scene_decode(clips / f"{names[0]}.mp4", det_paths[0], 11 * FPS),
        scene_cloud(coords, cams, 12 * FPS),
        scene_sweep(tree, coords, 11 * FPS),
        scene_end(5 * FPS),
    ]
    k = 0
    for gen in scenes:
        for img in gen:
            proc.stdin.write(img.tobytes())
            if k % 15 == 0:
                gif_frames.append(img.resize((432, 243), Image.LANCZOS))
            if k % FPS == 0:
                img.save(work / f"still_{k // FPS:03d}.jpg", quality=88)
            k += 1
    proc.stdin.close()
    proc.wait()
    gif = out_dir / "calibrate-the-tree.gif"
    gif_frames[0].save(gif, save_all=True, append_images=gif_frames[1:], duration=500, loop=0, optimize=True)
    print(f"{mp4}: {k / FPS:.1f} s, {mp4.stat().st_size / 1e6:.1f} MB; {gif}: {gif.stat().st_size / 1e6:.1f} MB")
    return mp4
