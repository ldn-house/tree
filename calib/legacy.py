"""The existing analyze.py + triangulate.py pipeline, wrapped for benchmarking.

Nothing here changes the behaviour of those scripts; it calls their functions
on in-memory detections so they can be scored against ground truth.
"""

import contextlib
import io

import numpy as np


def _quiet():
    return contextlib.redirect_stdout(io.StringIO())


def legacy_pipeline(view_uv: list[np.ndarray], sizes: list[tuple[int, int]], n: int = 500,
                    validate: bool = True, isotropic: bool = False, fix: bool = True) -> np.ndarray:
    """Run analyze --validate and triangulate (4-view multi-angle + fix_outliers).

    view_uv: per view (n, 2) pixel positions (NaN = no detection), in the
    order 0, 90, 180, 270 degrees. isotropic normalises x by the image height
    (the fix for review H8) instead of the width.
    """
    import analyze
    import triangulate

    pos_list = []
    with _quiet():
        for uv, (w, h) in zip(view_uv, sizes):
            positions = {i: (float(uv[i, 0]), float(uv[i, 1])) for i in range(n) if np.isfinite(uv[i]).all()}
            if validate:
                positions = analyze.validate_wire_constraint(positions, max_neighbor_dist=150.0)
                positions = analyze.interpolate_missing(positions, total_leds=n)
            pos = {}
            for i, (x, y) in positions.items():
                xn = 0.5 + (x - w / 2) / h if isotropic else x / w
                pos[str(i)] = {"x": x, "y": y, "x_norm": xn, "y_norm": y / h}
            pos_list.append({"positions": pos, "image_size": {"width": w, "height": h}})
        coords = triangulate.multi_angle_triangulate(pos_list, [f"v{k}" for k in range(len(pos_list))])
        if fix:
            coords = triangulate.fix_outliers(coords, threshold=3.0)
    out = np.full((n, 3), np.nan)
    for i, c in coords.items():
        out[i] = c
    return out
