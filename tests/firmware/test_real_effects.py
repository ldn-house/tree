"""The repo's effects/ package, bundled the way flash.py and CI build it.

The other device tests use the deliberately small fixtures/effects package
(fast and predictable); these check the real one."""

import base64
import json
import subprocess
import sys

import pytest

import devsim
from devsim import wait_for

sys.path.insert(0, str(devsim.ROOT))
import effects  # noqa: E402

FX = devsim.ROOT / "effects"


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Built with no --effects: the default must be the repo's effects/."""
    out = tmp_path_factory.mktemp("real")
    subprocess.run([sys.executable, str(devsim.ROOT / "tools" / "build_firmware.py"), "--out", str(out),
                    "--version", "real", "--generation", "1000"], check=True, capture_output=True)
    return out


def test_default_build_bundles_the_registered_effects_only(bundle):
    m = json.loads((bundle / "manifest.json").read_text())
    paths = {f["path"] for f in m["files"]}
    for name in effects.EFFECTS:
        assert f"effects/{name}.mpy" in paths, name
    for name in effects.DISABLED:
        assert f"effects/{name}.mpy" not in paths, name  # kept in the repo, not shipped
    assert {"effects/__init__.mpy", "effects/_output.mpy", "effects/_coords.mpy", "effects/_util.mpy"} <= paths
    assert m["bootmgr"] == 2


def test_every_effect_runs_on_the_device(make_device, bundle):
    d = make_device(bundle / "tree-firmware.bin", start=False)
    # Saved state, so the first effect also gets 2 s (set over HTTP, it would
    # keep the deadline it started with: its own DURATION_MS).
    (d.root / "state.json").write_text(json.dumps({"power": True, "effect": "auto", "auto_s": 2, "brightness": 255}))
    d.start()
    st = d.wait_http(40, pred=lambda s: not s["boot"]["trial"])
    assert st["effects"] == effects.EFFECTS
    assert st["coords"] > 0  # the bundled legacy coordinates
    assert st["state"]["auto_s"] == 2 and st["state"]["effect"] == "auto"
    seen = set()

    def all_seen():
        s = d.get("/")[1]
        seen.add(s["effect"]["running"])
        assert not s["effect"]["failed"], s["effect"]["failed"]
        return set(effects.EFFECTS) <= seen

    wait_for(all_seen, 2.5 * len(effects.EFFECTS) + 20, interval=0.3, what="every effect")
    assert len(d.boots) == 0


def test_new_coordinates_reach_a_cached_coordinate_effect(make_device, bundle):
    """effects/_util caches per-LED tables per coordinates list; after
    PUT /files/coords.txt the runtime hands effects a new list, so a
    coordinate effect renders from the new positions, not the cached ones."""
    d = make_device(bundle / "tree-firmware.bin")
    d.wait_http(40, pred=lambda s: not s["boot"]["trial"])
    n = 500

    def put(y):
        body = ("0.1,%.2f,0.1\n" % y * n).encode()
        code, res = d.request("PUT", "/files/coords.txt", body=body)
        assert code == 200 and res["known"] == n, res

    def lit():
        f = base64.b64decode(d.get("/frame")[1]["frame"])
        return sum(f)

    # plane_sweep: one up-and-down sweep every 5.3 s (speed 1) starting at
    # the bottom when the effect (re)starts, lighting LEDs within 0.03 of
    # the plane. With every LED at one height the whole frame lights only
    # when the plane passes it: ~0.1 s and ~5.2 s after the start for height
    # 0.05, ~2.5-2.8 s for 0.95. Stale cached tables would keep the old timing.
    import time

    def first_lit(y):
        put(y)  # reloads the effect: its clock starts again
        t0 = time.time()
        wait_for(lambda: d.get("/")[1]["effect"]["running"] == "plane_sweep", 10)
        time.sleep(0.5)  # past the first pass of a low plane
        while time.time() - t0 < 7:
            if lit() > 0:
                return time.time() - t0
            time.sleep(0.05)
        return None

    d.post("/state", {"effect": "plane_sweep", "brightness": 255, "speed": 1})
    low = first_lit(0.05)
    high = first_lit(0.95)
    low2 = first_lit(0.05)
    assert low is not None and 4.3 < low < 6.5, low
    assert high is not None and 1.8 < high < 3.6, high
    assert low2 is not None and 4.3 < low2 < 6.5, low2
    assert not d.get("/")[1]["effect"]["failed"]
