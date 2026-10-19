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


def test_views_from_two_cameras_get_their_own_focal_length():
    lay = get_layout("legacy-cleaned")
    cams = make_rig("phone", lay, 4, seed=5)
    for c in cams[2:]:  # a second phone with a longer lens
        c.fx = c.fy = c.fx * 1.3
    rng = np.random.default_rng(0)
    views = []
    for i, c in enumerate(cams):
        v = make_view(lay, c, SimParams(), rng)
        uv = v.uv + rng.normal(0, 1.0, v.uv.shape)
        uv[~v.visible] = np.nan
        views.append(ViewObs(c.name, c.width, c.height, uv, np.ones(lay.n), device=0 if i < 2 else 1))
    sol = Solver(views).solve()
    assert abs(sol.focal[0] / cams[0].fx - 1) < 0.04 and abs(sol.focal[1] / cams[2].fx - 1) < 0.04
    assert position_errors(sol.points, lay.points, lay.pitch)["median_cm"] < 1.0


def test_legacy_export_round_trips_through_adapt_legacy():
    # Mirrors effects/_coords.adapt_legacy (harness thread): undo the 1080/608
    # stretch, rescale y to [0, 1], centre x/z on their centroid.
    from calib.output import to_legacy
    rng = np.random.default_rng(1)
    norm = np.column_stack([rng.normal(0, 0.2, 50), rng.uniform(0, 1, 50), rng.normal(0, 0.2, 50)])
    norm[:, 1] = (norm[:, 1] - norm[:, 1].min()) / np.ptp(norm[:, 1])
    leg = to_legacy(norm)
    h = np.ptp(leg[:, 1])
    k = (608 / 1080) / h
    back = np.column_stack([(leg[:, 0] - leg[:, 0].mean()) * k, (leg[:, 1] - leg[:, 1].min()) / h,
                            (leg[:, 2] - leg[:, 2].mean()) * k])
    centred = norm - np.array([norm[:, 0].mean(), 0, norm[:, 2].mean()])
    assert np.allclose(back, centred, atol=1e-9)
