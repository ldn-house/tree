"""Runs tests/firmware/mp/test_units.py on the MicroPython unix port."""

import subprocess
from pathlib import Path

import devsim

HERE = Path(__file__).resolve().parent


def _calib_vector(tmp_path):
    """The calibration thread's test vector: in-tree once its branch lands,
    otherwise straight from that branch if this checkout has it."""
    tcal = devsim.ROOT / "tests" / "data" / "calib-default.tcal"
    sha = devsim.ROOT / "tests" / "data" / "calib-default.expanded.sha256"
    if tcal.exists() and sha.exists():
        return [str(tcal), str(sha)]
    out = []
    for name in ("calib-default.tcal", "calib-default.expanded.sha256"):
        r = subprocess.run(["git", "show", f"origin/calibration/video-pipeline:tests/data/{name}"],
                           cwd=devsim.ROOT, capture_output=True)
        if r.returncode != 0:
            return []
        p = tmp_path / name
        p.write_bytes(r.stdout)
        out.append(str(p))
    return out


def test_micropython_units(tmp_path):
    work = tmp_path / "flash"
    work.mkdir()
    r = subprocess.run([devsim.MICROPYTHON, str(HERE / "mp" / "test_units.py"), str(work), *_calib_vector(tmp_path)],
                       capture_output=True, text=True, timeout=120)
    print(r.stdout[-6000:], r.stderr[-2000:])
    assert r.returncode == 0, r.stdout[-3000:]
    assert " 0 failed" in r.stdout
