#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "numpy",
#     "scipy",
#     "opencv-python",
#     "matplotlib",
# ]
# ///
"""
Calibration toolkit: sequences, synthetic benchmark, detection, video decoding
and multi-view triangulation. See docs/calibration-findings.md.

Usage:
    # Write the standard sequence the firmware plays (docs/calibration-protocol.md)
    ./calibrate.py sequence --out sequences/default.tcal

    # Render a synthetic still capture (capture.py layout + truth.json)
    ./calibrate.py simulate --out .scratch/synth/webcam --rig webcam

    # Detect LEDs in captures (real or synthetic), one JSON per view
    ./calibrate.py detect --captures captures --views tue1,tue2,tue3,tue4 --lag 1

    # Triangulate into coordinates/coords.txt (front view first)
    ./calibrate.py solve positions/tue{1,2,3,4}_detections.json --out coordinates
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


def cmd_sequence(args):
    from calib.sequence import SequenceParams, build, sequential_plan

    if args.sequential:
        plan = sequential_plan(symbol_ms=args.symbol_ms)
    else:
        plan = build(SequenceParams(codebook=args.codebook, differential=not args.single,
                                    colour=args.colour, symbol_ms=args.symbol_ms,
                                    power_budget=args.power_budget))
    data = plan.to_tcal(loops=args.loops)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_bytes(data)
    power = plan.frame_power()
    print(f"{args.out}: {plan.n_frames} frames, loop {plan.loop_ms / 1000:.1f} s, "
          f"{len(data)} bytes, max frame power {power.max():.0f}/{args.power_budget}")


def cmd_simulate(args):
    import numpy as np
    from calib.layout import get_layout
    from calib.render import SimParams
    from calib.synth import make_rig, render_stills

    layout = get_layout(args.layout)
    cams = make_rig(args.rig, layout, args.views, seed=args.seed, scale=args.scale)
    params = SimParams(seed=args.seed)
    for kv in args.param or []:
        k, v = kv.split("=")
        setattr(params, k, type(getattr(params, k))(v) if not isinstance(getattr(params, k), str) else v)
    render_stills(layout, cams, params, Path(args.out), mode=args.mode, lag=args.lag,
                  repeat_p=args.repeat_p, skip_p=args.skip_p)
    print(f"wrote {args.views} views to {args.out}")


def cmd_detect(args):
    from calib.detect import detect_sequence, load_conf_model, save_detections

    model = load_conf_model()
    root = Path(args.captures)
    views = args.views.split(",") if args.views else sorted(
        p.name for p in root.iterdir() if (p / "sequential").is_dir())
    for v in views:
        res = detect_sequence(root / v / "sequential", args.method, lag=0 if args.raw_frames else args.lag,
                              conf_model=model, track_repeats=not args.raw_frames)
        out = Path(args.out) / f"{v}_detections.json"
        save_detections(res, out, v)
        n = sum(d is not None and d.conf >= 0.5 for d in res["detections"].values())
        print(f"{v}: {n}/500 confident detections, repeated frames {res['repeated_frames']} "
              f"(lag {res['lag']} -> {res['final_lag']}) -> {out}")


def cmd_bench_stills(args):
    import json
    from calib.bench import bench_stills
    from calib.detect import load_conf_model

    rows = []
    for d in args.data:
        rows += bench_stills(Path(d), methods=args.methods.split(","), model=load_conf_model())
    cols = ["dataset", "pipeline", "detector", "lag_assumed", "solved", "median_cm", "p90_cm",
            "within_5cm", "over_10cm", "id_swaps"]
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols))
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


def cmd_solve(args):
    import numpy as np
    from calib.detect import load_detections
    from calib.output import write_outputs
    from calib.solve import Solver, ViewObs, to_tree_frame

    paths = [Path(p) for p in args.detections]
    views = []
    for p in paths:
        d = load_detections(p)
        w, h = d["image_size"]
        views.append(ViewObs(d["view"], w, h, d["uv"], d["conf"], device=0 if args.same_camera else len(views)))
    sol = Solver(views, min_weight=args.min_conf, verbose=True).solve(wire_prior=not args.no_wire)
    norm, info = to_tree_frame(sol.points, sol, front_view=0)
    conf = write_outputs(Path(args.out), norm, sol, info, args.height_m, [v.name for v in views],
                         {"detections": [str(p) for p in paths]})
    print(f"median reprojection {sol.cost_px:.2f}px, focal {np.round(sol.focal)}, "
          f"views per LED {np.bincount(sol.views_used, minlength=5).tolist()}, "
          f"mean confidence {conf.mean():.2f} -> {args.out}/coords.txt")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sequence", help="write a .tcal calibration sequence")
    p.add_argument("--out", default="sequences/default.tcal")
    p.add_argument("--codebook", default="hamming",
                   choices=["binary", "gray", "gray_parity", "hamming"])
    p.add_argument("--colour", action="store_true", help="carry 3 bits per frame in R, G, B")
    p.add_argument("--single", action="store_true", help="no complement frames")
    p.add_argument("--sequential", action="store_true", help="one LED at a time")
    p.add_argument("--symbol-ms", type=int, default=100)
    p.add_argument("--loops", type=int, default=0, help="default loops (0 = until stopped)")
    p.add_argument("--power-budget", type=int, default=49725)
    p.set_defaults(func=cmd_sequence)

    p = sub.add_parser("simulate", help="render a synthetic still capture with ground truth")
    p.add_argument("--out", required=True)
    p.add_argument("--layout", default="legacy-cleaned", choices=["legacy-cleaned", "cone-spiral"])
    p.add_argument("--rig", default="webcam", choices=["webcam", "phone"])
    p.add_argument("--views", type=int, default=4)
    p.add_argument("--mode", default="sequential", choices=["sequential", "binary"])
    p.add_argument("--lag", type=int, default=0, help="simulate capture lag (frames)")
    p.add_argument("--repeat-p", type=float, default=0.0,
                   help="per-frame chance the camera returns the previous state again")
    p.add_argument("--skip-p", type=float, default=0.0, help="per-frame chance a state is never captured")
    p.add_argument("--scale", type=float, default=1.0, help="image size multiplier")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--param", action="append", help="override a SimParams field, e.g. ae=locked")
    p.set_defaults(func=cmd_simulate)

    p = sub.add_parser("detect", help="detect LEDs in sequential still captures")
    p.add_argument("--captures", default="captures")
    p.add_argument("--views", default=None, help="comma-separated view names (default: all)")
    p.add_argument("--method", default="diff",
                   choices=["ref_dog", "ref", "diff", "local_contrast", "background"])
    p.add_argument("--lag", type=int, default=0,
                   help="frame led_X shows LED X-lag (captures made before the fix: 1)")
    p.add_argument("--raw-frames", action="store_true",
                   help="key detections by frame, with no lag or repeat handling (input for solve --align)")
    p.add_argument("--out", default="positions")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("bench-stills", help="score detector x triangulation on synthetic stills")
    p.add_argument("data", nargs="+", help="synthetic capture directories (with truth.json)")
    p.add_argument("--methods", default="local_contrast,diff,ref,ref_dog")
    p.add_argument("--out", help="write rows as JSON")
    p.set_defaults(func=cmd_bench_stills)

    p = sub.add_parser("solve", help="triangulate detections into coords.txt (perspective BA)")
    p.add_argument("detections", nargs="+", help="*_detections.json, front view first")
    p.add_argument("--out", default="coordinates")
    p.add_argument("--height-m", type=float, default=1.75, help="lowest-to-highest LED height")
    p.add_argument("--min-conf", type=float, default=0.5)
    p.add_argument("--same-camera", action="store_true", default=True,
                   help="all views share intrinsics (default)")
    p.add_argument("--per-view-intrinsics", dest="same_camera", action="store_false")
    p.add_argument("--no-wire", action="store_true", help="disable the wire prior")
    p.set_defaults(func=cmd_solve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
