"""Board-sized heap campaign (see heapcheck.py). Needs a 32-bit MicroPython
unix build in $MICROPYTHON32 (CI builds one with MICROPY_FORCE_32BIT=1) and
the real effects: effects/ in the tree, or the harness branch."""

import json
import os
import shutil
import subprocess
import sys

import pytest

import devsim

HEAP = os.environ.get("HEAPCHECK_HEAP", "400K")  # Pico 2 W: 452,544 bytes free idle, minus Wi-Fi margin


def real_effects(tmp):
    if (devsim.ROOT / "effects" / "_output.py").exists():
        return devsim.ROOT / "effects"
    a = subprocess.run(["git", "archive", "origin/harness/effects-renderer", "effects"], cwd=devsim.ROOT,
                       capture_output=True)
    if a.returncode:
        return None
    subprocess.run(["tar", "-x", "-C", str(tmp)], input=a.stdout, check=True)
    return tmp / "effects"


@pytest.mark.skipif(not os.environ.get("MICROPYTHON32"), reason="needs a 32-bit MicroPython in $MICROPYTHON32")
def test_board_sized_heap_with_real_effects_and_tls(tmp_path):
    fx = real_effects(tmp_path)
    if fx is None:
        pytest.skip("real effects not available")
    key = "heap-signing-key"
    a = devsim.build_bundle(tmp_path / "a", effects=fx, key=key, version="heap-a", generation=100)
    b = devsim.build_bundle(tmp_path / "b", effects=fx, key=key, version="heap-b", generation=200)
    cert, pkey = tmp_path / "c.pem", tmp_path / "k.pem"
    subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", str(pkey), "-out",
                    str(cert), "-days", "1", "-subj", "/CN=127.0.0.1"], check=True, capture_output=True)
    tcal = None
    r = subprocess.run(["git", "show", "origin/calibration/video-pipeline:sequences/default.tcal"], cwd=devsim.ROOT,
                       capture_output=True)
    if r.returncode == 0:
        tcal = tmp_path / "default.tcal"
        tcal.write_bytes(r.stdout)
    out = tmp_path / "result.json"
    cmd = [sys.executable, str(devsim.HERE / "heapcheck.py"), "--micropython", os.environ["MICROPYTHON32"],
           "--bundle", str(a), "--update", str(b), "--key-string", key, "--cert", str(cert), "--key", str(pkey),
           "--heap", HEAP, "--root", str(tmp_path / "dev"), "--out", str(out)]
    if tcal:
        cmd += ["--calib", str(tcal)]
    mosquitto = os.environ.get("MOSQUITTO") or shutil.which("mosquitto") or "/opt/homebrew/opt/mosquitto/sbin/mosquitto"
    if os.path.exists(mosquitto):
        cmd += ["--mosquitto", mosquitto]
    elif os.environ.get("CI"):
        pytest.fail("mosquitto is needed for the MQTT part of the campaign")
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=1500)
    print(r.stdout[-4000:], r.stderr[-2000:])
    res = json.loads(out.read_text())
    assert res["ok"], res
