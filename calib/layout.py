"""LED layouts: loading coordinates, synthetic ground truth and quality metrics.

Units in this module are metres in a world frame with +y up and the trunk on
the y axis, unless a function says otherwise.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d

# analyze.py normalised x by the image width (608 px) and y by the height
# (1080 px), and triangulate.py treated both as one unit, so legacy x/z are
# stretched by 1080/608 relative to y (review H8).
LEGACY_ANISOTROPY = 608 / 1080


def load_coords(path: str | Path) -> np.ndarray:
    """Load coords.csv, coords_compact.txt or coords.txt into (n, 3), NaN for null."""
    path = Path(path)
    rows = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("index"):
            continue
        parts = line.split(",")
        if parts[0] == "null" or parts[-1] == "null":
            rows.append([np.nan] * 3)
        elif len(parts) >= 4 and path.suffix == ".csv":
            rows.append([float(v) for v in parts[1:4]])
        else:
            rows.append([float(v) for v in parts[:3]])
    return np.array(rows, float)


@dataclass
class TreeBody:
    """Foliage cone and trunk used for occlusion (metres, trunk on the y axis)."""
    height: float           # top of foliage
    base_radius: float      # foliage radius at y = 0
    bottom: float = 0.0     # lowest foliage
    trunk_radius: float = 0.035
    density: float = 6.0    # optical depth per metre inside the foliage

    def radius(self, y):
        t = np.clip((y - self.bottom) / (self.height - self.bottom), 0, 1)
        return self.base_radius * (1 - t)


@dataclass
class Layout:
    name: str
    points: np.ndarray      # (n, 3) metres
    body: TreeBody
    pitch: float            # wire length between consecutive LEDs (m)

    @property
    def n(self) -> int:
        return len(self.points)


def relax_wire(p: np.ndarray, pitch: float, iters: int = 200) -> np.ndarray:
    """Project consecutive distances onto <= pitch (position-based dynamics)."""
    p = p.copy()
    for _ in range(iters):
        d = np.diff(p, axis=0)
        L = np.linalg.norm(d, axis=1)
        over = L > pitch
        if not over.any():
            break
        corr = np.zeros_like(d)
        corr[over] = d[over] * ((L[over] - pitch) / L[over])[:, None] * 0.5
        p[:-1] += corr
        p[1:] -= corr
    return p


def fit_body(points: np.ndarray, inner: float = 0.85) -> TreeBody:
    y = points[:, 1]
    r = np.hypot(points[:, 0], points[:, 2])
    lo, hi = np.nanmin(y), np.nanmax(y)
    # Fit r_max(y) = R0 (1 - (y - lo) / (top - lo)) through high-percentile radii.
    bins = np.linspace(lo, hi, 9)
    ys, rs = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        m = (y >= a) & (y < b)
        if m.sum() > 5:
            ys.append((a + b) / 2)
            rs.append(np.percentile(r[m], 90))
    A = np.vstack([np.ones(len(ys)), ys]).T
    c0, c1 = np.linalg.lstsq(A, np.array(rs), rcond=None)[0]
    top = -c0 / c1 if c1 < 0 else hi * 1.1
    R0 = c0 + c1 * lo
    return TreeBody(height=max(top, hi + 0.05), base_radius=R0 * inner, bottom=lo - 0.1)


# The December 2025 coordinates (triangulate.py output), frozen here as the seed
# for the synthetic ground truth, so recalibrating the tree does not change the
# benchmark.
LEGACY_COORDS = Path(__file__).with_name("data") / "legacy-coords-2025-12.csv"


def cleaned_from_legacy(path: str | Path = LEGACY_COORDS, height_m: float = 1.75,
                        pitch: float = 0.10, seed: int = 0) -> Layout:
    """A plausible ground truth derived from the current (noisy) coordinates.

    1. Undo the legacy x/z stretch and scale to height_m.
    2. Smooth along the wire to remove per-LED detection jitter (sigma 1.5).
    3. Add branch-scale in/out zigzag, as a real string wrapped through branches.
    4. Relax so no consecutive pair is further apart than the wire pitch.
    """
    rng = np.random.default_rng(seed)
    p = load_coords(path)
    p[:, [0, 2]] *= LEGACY_ANISOTROPY
    p[:, 1] -= np.nanmin(p[:, 1])
    p *= height_m / np.nanmax(p[:, 1])
    centre = np.nanmedian(p[:, [0, 2]], axis=0)
    p[:, [0, 2]] -= centre
    p = gaussian_filter1d(p, 1.5, axis=0, mode="nearest")
    # Zigzag: correlated radial excursion plus a little tangential/vertical wander.
    r_hat = p[:, [0, 2]] / np.maximum(np.hypot(p[:, 0], p[:, 2]), 1e-6)[:, None]
    radial = gaussian_filter1d(rng.normal(0, 1, len(p)), 1.2) * 0.12
    wander = gaussian_filter1d(rng.normal(0, 1, (len(p), 3)), 1.0, axis=0) * 0.03
    p[:, 0] += r_hat[:, 0] * radial
    p[:, 2] += r_hat[:, 1] * radial
    p += wander
    p = relax_wire(p, pitch)
    p[:, 1] -= p[:, 1].min()
    return Layout("legacy-cleaned", p, fit_body(p), pitch)


def cone_spiral(n: int = 500, height: float = 1.75, base_radius: float = 0.55,
                turns: float = 9.0, pitch: float = 0.10, seed: int = 1) -> Layout:
    """Procedural layout: string wound bottom-up around a cone, dipping into branches."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, n) ** 0.8          # more LEDs low down, where the tree is wide
    y = t * height * 0.97
    r = base_radius * (1 - y / height)
    depth = np.clip(gaussian_filter1d(rng.normal(0, 1, n), 1.0) * 0.35 + 0.15, -0.2, 0.7)
    r = r * (1 - 0.6 * np.clip(depth, 0, 1)) + 0.02
    theta = 2 * np.pi * turns * t + gaussian_filter1d(rng.normal(0, 0.3, n), 2)
    p = np.stack([r * np.cos(theta), y, r * np.sin(theta)], 1)
    p += gaussian_filter1d(rng.normal(0, 0.015, (n, 3)), 1, axis=0)
    p = relax_wire(p, pitch)
    p[:, 1] -= p[:, 1].min()
    return Layout("cone-spiral", p, TreeBody(height=height + 0.08, base_radius=base_radius * 0.9,
                                             bottom=-0.08), pitch)


def get_layout(name: str, **kw) -> Layout:
    if name == "legacy-cleaned":
        return cleaned_from_legacy(**kw)
    if name == "cone-spiral":
        return cone_spiral(**kw)
    raise ValueError(name)


def quality(points: np.ndarray, pitch: float | None = None, dup_eps: float | None = None) -> dict:
    """Self-consistency metrics for a coordinate set, without ground truth.

    Distances are in the coordinates' own units. If pitch is not given, it is
    estimated as the median consecutive distance (an overestimate when the
    coordinates are noisy).
    """
    p = np.asarray(points, float)
    valid = ~np.isnan(p).any(1)
    n = len(p)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    seg_ok = valid[:-1] & valid[1:]
    seg = seg[seg_ok]
    med = float(np.median(seg))
    pitch_used = pitch if pitch is not None else med
    eps = dup_eps if dup_eps is not None else 0.05 * pitch_used
    q = p[valid]
    d = np.linalg.norm(q[:, None] - q[None], axis=-1)
    iu = np.triu_indices(len(q), 1)
    dup_pairs = int((d[iu] < eps).sum())
    exact = len(q) - len(np.unique(np.round(q, 6), axis=0))
    # Outlier: far from the midpoint of its two neighbours, relative to the pitch.
    mid_dev = np.full(n, np.nan)
    for i in range(1, n - 1):
        if valid[i - 1] and valid[i] and valid[i + 1]:
            mid_dev[i] = np.linalg.norm(p[i] - (p[i - 1] + p[i + 1]) / 2)
    y = p[valid, 1]
    r = np.hypot(p[valid, 0] - np.median(p[valid, 0]), p[valid, 2] - np.median(p[valid, 2]))
    return {
        "n_valid": int(valid.sum()),
        "pitch_used": pitch_used,
        "step_median": med,
        "step_p95": float(np.percentile(seg, 95)),
        "step_max": float(seg.max()),
        "continuity_violations": int((seg > 1.5 * pitch_used).sum()),
        "continuity_violations_2x": int((seg > 2 * pitch_used).sum()),
        "exact_duplicates": int(exact),
        "near_duplicate_pairs": dup_pairs,
        "outliers": int(np.nansum(mid_dev > 1.5 * pitch_used)),
        "path_length": float(seg.sum()),
        "height": float(np.ptp(y)),
        "height_over_width": float(np.ptp(y) / (2 * np.percentile(r, 98))),
        "corr_index_height": float(np.corrcoef(np.arange(n)[valid], y)[0, 1]),
    }
