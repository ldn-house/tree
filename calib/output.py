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
