"""Scoring against ground truth, and real-data consistency metrics."""

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from .detect import FEATURES, detect_sequence
from .synth import load_truth

CORRECT_PX = 4.0   # at 1080 px long side; scaled with image size
GROSS_PX = 10.0


def score_view(result: dict, uv_true: np.ndarray, visible: np.ndarray, min_conf: float = 0.5) -> dict:
    n = len(uv_true)
    scale = max(result["image_size"]) / 1080
    dets = result["detections"]
    err = np.full(n, np.nan)
    conf = np.zeros(n)
    for i in range(n):
        d = dets.get(i)
        if d is not None and np.isfinite(uv_true[i]).all():
            err[i] = np.hypot(d.x - uv_true[i, 0], d.y - uv_true[i, 1]) / scale
            conf[i] = d.conf
        elif d is not None:
            err[i] = np.inf
            conf[i] = d.conf
    reported = conf >= min_conf
    correct = reported & (err <= CORRECT_PX)
    gross = reported & (err > GROSS_PX)
    return {
        "visible": int(visible.sum()),
        "reported": int(reported.sum()),
        "correct": int(correct.sum()),
        "recall_visible": float((correct & visible).sum() / max(visible.sum(), 1)),
        "precision": float(correct.sum() / max(reported.sum(), 1)),
        "gross_reported": int(gross.sum()),
        "median_err_px": float(np.median(err[correct])) if correct.any() else float("nan"),
        "p90_err_px": float(np.percentile(err[correct], 90)) if correct.any() else float("nan"),
    }


def _run(args):
    root, view, method, lag, model = args
    return view, detect_sequence(Path(root) / view / "sequential", method, lag=lag, conf_model=model)


def run_detectors(root: Path, methods: list[str], model: dict | None, lag: int = 0, workers: int = 8):
    truth = load_truth(root / "truth.json")
    views = [v["camera"].name for v in truth["views"]]
    jobs = [(str(root), v, m, lag, model) for m in methods for v in views]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_run, jobs))
    out = {}
    for (rootv, view, method, _l, _m), (_v, res) in zip(jobs, results):
        out[(method, view)] = res
    return truth, views, out


def feature_table(res: dict, uv_true: np.ndarray):
    X, y = [], []
    scale = max(res["image_size"]) / 1080
    for i, d in res["detections"].items():
        if d is None or not np.isfinite(uv_true[i]).all():
            if d is not None:
                X.append(d.features())
                y.append(0)
            continue
        X.append(d.features())
        y.append(float(np.hypot(d.x - uv_true[i, 0], d.y - uv_true[i, 1]) / scale <= CORRECT_PX))
    return np.array(X), np.array(y)


def wire_jumps(uv: np.ndarray, scale_px: float) -> dict:
    """2D wire continuity for real data: jumps between consecutive detected LEDs."""
    ok = np.isfinite(uv).all(1)
    pairs = ok[:-1] & ok[1:]
    d = np.linalg.norm(np.diff(uv, axis=0), axis=1)[pairs]
    return {"pairs": int(pairs.sum()), "median_step_px": float(np.median(d)),
            "jumps_gt_scale": int((d > scale_px).sum()),
            "frac_jumps": float((d > scale_px).mean())}


def _uv_from(res: dict, n: int, min_conf: float = 0.5):
    uv = np.full((n, 2), np.nan)
    w = np.zeros(n)
    for i, d in res["detections"].items():
        if d is not None and d.conf >= min_conf:
            uv[i] = (d.x, d.y)
            w[i] = d.conf
    return uv, w


def solve_detections(per_view: list[tuple[np.ndarray, np.ndarray]], sizes, names, wire_prior=True):
    from .solve import Solver, ViewObs
    views = [ViewObs(nm, w, h, uv, wt) for (uv, wt), (w, h), nm in zip(per_view, sizes, names)]
    return Solver(views).solve(wire_prior=wire_prior)


def bench_stills(root: Path, methods=("local_contrast", "diff", "ref", "ref_dog"), lag: int | None = None,
                 model: dict | None = None) -> list[dict]:
    """End-to-end: detector x triangulation, scored in 3D against truth."""
    from .legacy import legacy_pipeline
    from .metrics import position_errors

    truth = load_truth(root / "truth.json")
    true_lag = truth.get("lag", 0) if lag is None else lag
    n = len(truth["points"])
    pitch = truth["pitch"]
    names = [v["camera"].name for v in truth["views"]]
    sizes = [(v["camera"].width, v["camera"].height) for v in truth["views"]]
    rows = []
    lags = {0} | {true_lag}
    jobs = {(m, L) for m in methods for L in lags}
    dets = {}
    with ProcessPoolExecutor(8) as ex:
        futs = {(m, L, v): ex.submit(_run, (str(root), v, m, L, model)) for (m, L) in jobs for v in names}
        for k, f in futs.items():
            dets[k] = f.result()[1]

    def add(pipeline, method, L, coords, extra=None, reflect=False):
        try:
            r = position_errors(coords, truth["points"], pitch, allow_reflection=reflect)
        except ValueError:
            r = {"solved": int(np.isfinite(coords).all(1).sum()), "median_cm": float("nan"),
                 "p90_cm": float("nan"), "within_5cm": 0.0, "over_10cm": n, "id_swaps": n}
        rows.append({"dataset": root.name, "pipeline": pipeline, "detector": method, "lag_assumed": L,
                     **r, **(extra or {})})

    for m in methods:
        for L in sorted(lags):
            per_view = [_uv_from(dets[(m, L, v)], n) for v in names]
            if m == "local_contrast":
                add("legacy (analyze --validate + multi_angle)", m, L,
                    legacy_pipeline([p[0] for p in per_view], sizes, n), reflect=True)
                add("legacy, isotropic pixels", m, L,
                    legacy_pipeline([p[0] for p in per_view], sizes, n, isotropic=True), reflect=True)
            try:
                sol = solve_detections(per_view, sizes, names)
            except RuntimeError as e:
                add("perspective BA + wire prior", m, L, np.full((n, 3), np.nan), {"error": str(e)})
                continue
            add("perspective BA + wire prior", m, L, sol.points,
                {"reproj_px": sol.cost_px, "focal": [round(f) for f in sol.focal]})
            if m == "diff" and L == true_lag:
                sol2 = solve_detections(per_view, sizes, names, wire_prior=False)
                add("perspective BA, no wire prior", m, L, sol2.points, {"reproj_px": sol2.cost_px})
    return rows


def _decode_job(args):
    from .sequence import SequenceParams, build, sequential_plan
    from .video import decode_video
    path, plan_kw = args
    sequential = plan_kw.pop("sequential", False)
    plan = sequential_plan(symbol_ms=plan_kw["symbol_ms"]) if sequential else build(SequenceParams(**plan_kw))
    return decode_video(Path(path), plan)


def bench_video(root: Path, workers: int = 4) -> dict:
    """Decode every clip in a simulate-video directory and score 2D, ID and 3D accuracy."""
    import json
    from .metrics import position_errors
    from .solve import Solver, ViewObs

    truth = json.loads((root / "truth.json").read_text())
    plan_kw = dict(truth["plan"])
    names = [v["camera"]["name"] for v in truth["views"]]
    with ProcessPoolExecutor(workers) as ex:
        results = list(ex.map(_decode_job, [(str(root / f"{nm}.mp4"), dict(plan_kw)) for nm in names]))
    n = len(truth["points"])
    per_view, scores, views = [], [], []
    for nm, tv, res in zip(names, truth["views"], results):
        uv_true = np.array(tv["uv"], float)
        uv_true[uv_true[:, 0] < 0] = np.nan
        vis = np.array(tv["visible"], bool)
        sc = score_view(res, uv_true, vis)
        # Wrong IDs: confident detections far from their own LED but close to another.
        uv, w = _uv_from(res, n)
        ok = np.isfinite(uv).all(1)
        d_own = np.linalg.norm(uv - uv_true, axis=1)
        sc["wrong_id"] = int((ok & ~(d_own <= GROSS_PX * max(res["image_size"]) / 1080)).sum())
        sc["sync"] = res["sync"]
        scores.append(sc)
        w_img, h_img = res["image_size"]
        views.append(ViewObs(nm, w_img, h_img, uv, w))
    out = {"dataset": root.name, "plan": plan_kw, "views": scores}
    usable = [v for v in views if np.isfinite(v.uv).all(1).sum() >= 15]
    out["views_used_3d"] = len(usable)
    try:
        if len(usable) < 2:
            raise RuntimeError(f"only {len(usable)} views with 15+ detections")
        sol = Solver(usable).solve()
        out["3d"] = position_errors(sol.points, np.array(truth["points"]), truth["pitch"])
        out["3d"]["reproj_px"] = sol.cost_px
    except RuntimeError as e:
        out["3d"] = {"error": str(e)}
    return out


def summarise_video(r: dict) -> str:
    v = r["views"]
    tot = lambda k: sum(x[k] for x in v)
    d3 = r["3d"]
    s3 = (f"3D median {d3['median_cm']:.2f}cm p90 {d3['p90_cm']:.2f} >10cm {d3['over_10cm']} swaps {d3['id_swaps']}"
          if "median_cm" in d3 else f"3D failed: {d3.get('error')}")
    rec = sum(x["recall_visible"] * x["visible"] for x in v) / max(tot("visible"), 1)
    return (f"{r['dataset']}: visible {tot('visible')} correct {tot('correct')} "
            f"recall {rec:.3f} wrong-ID {tot('wrong_id')} "
            f"median2D {np.nanmean([x['median_err_px'] for x in v]):.2f}px | {s3}")
