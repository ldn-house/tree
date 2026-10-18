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


def _views(seed=5, dup=False):
    lay = get_layout("legacy-cleaned")
    cams = make_rig("phone", lay, 4, seed=seed)
    rng = np.random.default_rng(0)
    views = []
    for c in cams:
        v = make_view(lay, c, SimParams(), rng)
        uv = v.uv + rng.normal(0, 1.0, v.uv.shape)
        uv[~v.visible] = np.nan
        views.append(ViewObs(c.name, c.width, c.height, uv, np.ones(lay.n)))
    if dup:
        # A second clip from (almost) the same spot as the first, sharing every LED.
        uv = views[0].uv + rng.normal(0, 1.0, views[0].uv.shape) + np.array([6.0, 2.0])
        views.insert(1, ViewObs("view0b", views[0].width, views[0].height, uv, np.ones(lay.n)))
    return lay, cams, views


def test_focal_length_converges_from_a_wrong_guess():
    # Bundle adjustment used to leave the focal length at its initial guess.
    lay, cams, views = _views()
    sol = Solver(views).solve(f_guesses=(1.2,))  # 2304 px against a true 1285
    assert abs(sol.focal[0] / cams[0].fx - 1) < 0.03
    assert position_errors(sol.points, lay.points, lay.pitch)["median_cm"] < 1.0


def test_near_duplicate_viewpoints_are_flagged_and_handled():
    # The real failure: two clips from the same spot shared the most LEDs, so they
    # were chosen to start from, which pins down no depth.
    lay, cams, views = _views(dup=True)
    solver = Solver(views)
    dupes = solver.duplicate_viewpoints()
    assert [(a, b) for a, b, _s, _n in dupes] == [("view0", "view0b")]
    sol = solver.solve()
    assert sol.views == [v.name for v in views] and not sol.dropped
    err = position_errors(sol.points, lay.points, lay.pitch)
    assert err["median_cm"] < 1.0 and err["over_10cm"] <= 5
