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


def _run_regressions(firmware, tmp_path):
    work = tmp_path / ("rr-" + firmware.parent.name)
    work.mkdir()
    return subprocess.run([devsim.MICROPYTHON, str(HERE / "mp" / "regress_review1.py"), str(firmware), str(work)],
                          capture_output=True, text=True, timeout=60)


def test_review_round1_reproductions_are_fixed(tmp_path):
    r = _run_regressions(devsim.ROOT / "firmware", tmp_path)
    print(r.stdout)
    assert r.returncode == 0 and "FAIL" not in r.stdout, r.stdout + r.stderr


def test_review_round1_reproductions_fail_on_the_reviewed_commit(tmp_path):
    """The same script against 57c4ce1 (the commit the review covered) must
    still fail, or it is not testing anything."""
    old = tmp_path / "old"
    old.mkdir()
    a = subprocess.run(["git", "archive", "57c4ce1", "firmware"], cwd=devsim.ROOT, capture_output=True)
    if a.returncode:
        import pytest

        pytest.skip("reviewed commit not in this checkout")
    subprocess.run(["tar", "-x", "-C", str(old)], input=a.stdout, check=True)
    r = _run_regressions(old / "firmware", tmp_path)
    print(r.stdout)
    for finding in ("B1", "H2", "H3 rejected", "H3 dry-run", "H5"):
        assert "FAIL " + finding in r.stdout, finding


def test_m8_mqtt_drop_closes_the_socket(tmp_path):
    """Review M8 (fw21-review-streams.py): Stream.close() is a no-op in
    MicroPython, so after _drop() the peer must see EOF without anyone
    awaiting wait_closed()."""
    import socket
    import threading

    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    seen = []

    def peer():
        c, _ = srv.accept()
        c.settimeout(5)
        try:
            while True:  # DISCONNECT, then EOF
                b = c.recv(100)
                seen.append(b)
                if not b:
                    break
        except OSError as e:
            seen.append(e)
        c.close()

    t = threading.Thread(target=peer)
    t.start()
    script = tmp_path / "drop.py"
    script.write_text(f"""import sys
sys.path.insert(0, {str(devsim.ROOT / 'firmware')!r})
import asyncio
from app.mqtt import MQTT
async def main():
    r, w = await asyncio.open_connection('127.0.0.1', {port})
    m = MQTT('127.0.0.1', {port}, 'review')
    m.r = m.w = w
    m.connected = True
    m._drop('review close check')
    assert m.r is None and m.w is None
    await asyncio.sleep_ms(500)  # no wait_closed(): the peer must already see EOF
asyncio.run(main())
""")
    r = subprocess.run([devsim.MICROPYTHON, str(script)], capture_output=True, text=True, timeout=20)
    t.join(10)
    srv.close()
    assert r.returncode == 0, r.stdout + r.stderr
    assert b"".join(x for x in seen if isinstance(x, bytes)) == b"\xe0\x00" and seen[-1] == b"", seen


def _run_regressions2(firmware, tmp_path):
    work = tmp_path / ("rr2-" + firmware.parent.name)
    work.mkdir()
    return subprocess.run([devsim.MICROPYTHON, str(HERE / "mp" / "regress_review2.py"), str(firmware), str(work)],
                          capture_output=True, text=True, timeout=120)


def test_review_round2_reproductions_are_fixed(tmp_path):
    r = _run_regressions2(devsim.ROOT / "firmware", tmp_path)
    print(r.stdout)
    assert r.returncode == 0 and "FAIL" not in r.stdout, r.stdout + r.stderr


def test_review_round2_reproductions_fail_on_the_reviewed_commit(tmp_path):
    """The same script against 1d0b869 (the commit round 2 reviewed)."""
    old = tmp_path / "old"
    old.mkdir()
    a = subprocess.run(["git", "archive", "1d0b869", "firmware"], cwd=devsim.ROOT, capture_output=True)
    if a.returncode:
        import pytest

        pytest.skip("reviewed commit not in this checkout")
    subprocess.run(["tar", "-x", "-C", str(old)], input=a.stdout, check=True)
    r = _run_regressions2(old / "firmware", tmp_path)
    print(r.stdout)
    for finding in ("R2-H1 config with HTTP_PORT null", "R2-H1 unconfirmed", "R2-H2", "R2-H3", "R2-M4 floor after",
                    "R2-M4 floor of", "R2-M5", "R2-M6", "R2-M8", "R2-L7"):
        assert "FAIL " + finding in r.stdout, finding
