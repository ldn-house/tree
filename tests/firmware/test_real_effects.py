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
    auto = [n for n in effects.EFFECTS if getattr(effects.load(n), "AUTO", True)]
    manual = [n for n in effects.EFFECTS if n not in auto]
    listed = {e["name"]: e["auto"] for e in d.get("/effects")[1]["effects"]}
    assert listed == {n: n in auto for n in effects.EFFECTS}
    seen = set()

    def all_seen():
        s = d.get("/")[1]
        seen.add(s["effect"]["running"])
        assert not s["effect"]["failed"], s["effect"]["failed"]
        return set(auto) <= seen

    wait_for(all_seen, 2.5 * len(auto) + 20, interval=0.3, what="every auto-cycled effect")
    assert not seen & set(manual), "AUTO = False effects in the auto cycle"
    # The rest run when selected.
    for name in manual:
        assert d.post("/state", {"effect": name})[0] == 200
        wait_for(lambda: d.get("/")[1]["effect"]["running"] == name, 10, what=name)
        wait_for(lambda: d.get("/")[1]["effect"]["fps"] > 0, 10, what=name + " frames")
        assert not d.get("/")[1]["effect"]["failed"]
    assert len(d.boots) == 0


def test_new_coordinates_reach_a_cached_coordinate_effect(make_device, bundle):
    """effects/_util caches per-LED tables per coordinates list; after
    PUT /files/coords.txt the runtime hands effects a new list, so a
    coordinate effect renders from the new positions, not the cached ones."""
    d = make_device(bundle / "tree-firmware.bin")
    d.wait_http(40, pred=lambda s: not s["boot"]["trial"])
    # Spread the known LEDs over every height: some are always inside the
    # plane, while unknown LEDs must stay black. Swap the two halves to
    # prove both that new coordinates arrive and old cached tables go away.
    d.post("/state", {"effect": "plane_sweep", "brightness": 255,
                      "transition_style": "none"})
    for half in (0, 1, 0):
        known = ["0.1,%.4f,0.1\n" % (i / 249) for i in range(250)]
        unknown = ["null\n"] * 250
        body = "".join(known + unknown if half == 0 else unknown + known).encode()
        code, res = d.request("PUT", "/files/coords.txt", body=body)
        assert code == 200 and res["known"] == 250, res

        def updated():
            f = base64.b64decode(d.get("/frame")[1]["frame"])
            halves = (sum(f[:750]), sum(f[750:]))
            return halves[half] > 0 and halves[1 - half] == 0

        wait_for(updated, 10, what="wire frame uses the new known positions")
        assert not d.get("/")[1]["effect"]["failed"]
