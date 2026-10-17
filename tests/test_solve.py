"""Perspective bundle adjustment on ideal synthetic projections."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from calib.layout import get_layout  # noqa: E402
from calib.metrics import position_errors  # noqa: E402
from calib.render import SimParams, make_view  # noqa: E402
from calib.solve import Solver, ViewObs, to_tree_frame  # noqa: E402
from calib.synth import make_rig  # noqa: E402


def test_solver_recovers_layout_and_frame():
    lay = get_layout("legacy-cleaned")
    cams = make_rig("webcam", lay, 4, seed=5)
    rng = np.random.default_rng(0)
    views = []
    for c in cams:
        v = make_view(lay, c, SimParams(), rng)
        uv = v.uv + rng.normal(0, 1.0, v.uv.shape)
        uv[~v.visible] = np.nan
        views.append(ViewObs(c.name, c.width, c.height, uv, np.ones(lay.n)))
    sol = Solver(views).solve()
    err = position_errors(sol.points, lay.points, lay.pitch)
    assert err["median_cm"] < 1.0 and not err["reflected"]
    norm, info = to_tree_frame(sol.points, sol)
    assert abs(np.nanmin(norm[:, 1])) < 1e-9 and abs(np.nanmax(norm[:, 1]) - 1) < 1e-9
    front = np.array(info["cameras"][0])
    assert front[0] > 0 and abs(front[2]) < 1e-6  # +x towards the front camera
