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

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
