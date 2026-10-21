# Calibrated LED coordinates

| File | What it is | Who reads it |
|---|---|---|
| `coords.txt` | **The calibrated coordinates.** One `x,y,z` (or `null`) line per LED in the contract frame: y from 0 (lowest LED) to 1 (highest), x and z in the same units, origin on the trunk axis, +x towards the front view, right-handed. | The firmware rewrite (installed as `/coords.txt`, preferred over the legacy file), `effects/_coords.py`. |
| `coords_metres.csv` | The same positions in metres (assuming a 1.75 m light span), with per-LED confidence, the number of views that saw it and its reprojection error. LEDs seen in fewer than two views were placed along the wire and have low confidence. | People and tools that want real units or confidence. |
| `calibration_report.json` | How the solve went: views used, focal length and distortion, median reprojection error, camera positions in the tree frame, near-duplicate viewpoints, and the solver log. No image data. | Debugging and re-solving. |
| `coords_compact.txt`, `coords.json` | **Generated from `coords.txt`** in the legacy frame of `triangulate.py` (y 0–1.86, x/z stretched by 1080/608), by `./calibrate.py export-legacy`. Kept so the firmware on main, `flash.py --coords`, the old simulator and `effects/_coords.adapt_legacy()` keep working with the new positions. | Legacy consumers only. Don't edit them by hand; regenerate. |

The current files come from the first video calibration (18 October 2026): five phone clips taken by turning the tree in front of a fixed phone, decoded and solved with `calibrate.py` (median reprojection 0.65 px, 453 of 500 LEDs seen at least once). See [docs/calibration-findings.md](../docs/calibration-findings.md).

To recalibrate, follow [docs/calibrate-the-tree.md](../docs/calibrate-the-tree.md), then:

```bash
./calibrate.py solve positions/tonight/*_detections.json --out coordinates
./calibrate.py export-legacy coordinates/coords.txt
```

The December 2025 coordinates that used to live here (`coords.csv`) were index-scrambled by a capture-timing bug. They're kept only as the seed for the synthetic benchmark, at `calib/data/legacy-coords-2025-12.csv`.
