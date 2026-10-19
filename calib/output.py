"""Writing coordinates in the shared contract (docs/effect-api.md, "Coordinates")."""

import json
from pathlib import Path

import numpy as np


def led_confidence(views_used: np.ndarray, reproj_px: np.ndarray) -> np.ndarray:
    """Heuristic per-LED confidence: views seen in, discounted by reprojection error."""
    base = np.select([views_used >= 3, views_used == 2, views_used == 1], [1.0, 0.8, 0.4], 0.1)
    penalty = np.exp(-np.nan_to_num(reproj_px, nan=0.0) / 6.0)
    return np.round(base * np.where(views_used >= 2, penalty, 1.0), 3)


def write_coords_txt(path: Path, norm: np.ndarray):
    """One `x,y,z` (or `null`) line per LED: y in [0, 1], x/z same units, origin on the trunk."""
    lines = ["null" if not np.isfinite(p).all() else f"{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}" for p in norm]
    Path(path).write_text("\n".join(lines) + "\n")


def write_outputs(out_dir: Path, norm: np.ndarray, sol, info: dict, height_m: float, names: list[str],
                  extra: dict | None = None):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_coords_txt(out_dir / "coords.txt", norm)
    conf = led_confidence(sol.views_used, sol.reproj_px)
    metres = norm * height_m
    rows = ["index,x_m,y_m,z_m,confidence,views,reproj_px"]
    for i, p in enumerate(metres):
        if np.isfinite(p).all():
            rows.append(f"{i},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f},{conf[i]:.3f},{sol.views_used[i]},"
                        f"{'' if not np.isfinite(sol.reproj_px[i]) else round(float(sol.reproj_px[i]), 2)}")
        else:
            rows.append(f"{i},,,,0,0,")
    (out_dir / "coords_metres.csv").write_text("\n".join(rows) + "\n")
    report = {
        "frame": "y up, y=0 lowest LED, y=1 highest LED; x,z same units; origin on trunk axis; "
                 "+x towards the first (front) view; right-handed",
        "height_m_assumed": height_m,
        "views": names,
        "focal_px": [float(f) for f in sol.focal],
        "distortion": [list(map(float, k)) for k in sol.dist],
        "median_reproj_px": sol.cost_px,
        "cameras_tree_frame": info["cameras"],
        "views_per_led": np.bincount(sol.views_used, minlength=5).tolist(),
        "log": sol.log,
        **(extra or {}),
    }
    (out_dir / "calibration_report.json").write_text(json.dumps(report, indent=1))
    return conf


# The legacy frame (triangulate.py's coords_compact.txt / coords.json): y from 0
# up to about 1.86 and x/z stretched by 1080/608 (review H8). The firmware on
# main and the old simulator still read it, and effects/_coords.adapt_legacy()
# maps it back into the contract frame.
LEGACY_HEIGHT = 1.859
LEGACY_XZ_SCALE = 608 / 1080


def to_legacy(norm: np.ndarray) -> np.ndarray:
    """Contract-frame coordinates -> the legacy frame, the inverse of adapt_legacy.

    x/z are re-centred on their centroid, as adapt_legacy expects, so the round
    trip is exact; in the contract frame the centroid is within a few mm of the
    trunk axis.
    """
    p = np.array(norm, float)
    ok = np.isfinite(p).all(1)
    c = p[ok][:, [0, 2]].mean(0)
    out = np.full_like(p, np.nan)
    k = LEGACY_HEIGHT / LEGACY_XZ_SCALE
    out[ok, 0] = (p[ok, 0] - c[0]) * k
    out[ok, 1] = p[ok, 1] * LEGACY_HEIGHT
    out[ok, 2] = (p[ok, 2] - c[1]) * k
    return out


def write_legacy(out_dir: Path, norm: np.ndarray):
    """coords_compact.txt and coords.json in the legacy frame, from contract coordinates."""
    leg = to_legacy(norm)
    lines = ["null" if not np.isfinite(q).all() else f"{q[0]:.4f},{q[1]:.4f},{q[2]:.4f}" for q in leg]
    (Path(out_dir) / "coords_compact.txt").write_text("\n".join(lines) + "\n")
    data = {
        "led_count": len(leg),
        "found_count": int(np.isfinite(leg).all(1).sum()),
        "coordinate_system": {"x": "left-right", "y": "up (0 = bottom)", "z": "front-back"},
        "range": "legacy frame: y 0-1.86, x/z stretched by 1080/608; generated from coords.txt",
        "coordinates": [None if not np.isfinite(q).all() else {"x": float(q[0]), "y": float(q[1]), "z": float(q[2])}
                        for q in leg],
    }
    (Path(out_dir) / "coords.json").write_text(json.dumps(data, indent=2) + "\n")
