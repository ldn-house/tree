"""OTA onto an older recovery layer (found on hardware).

The recovery layer is only installed over USB, so an OTA-installed app runs
on whatever recovery layer the board has. 7bacc8b's app called bootmgr
functions that 52cda75's recovery layer lacks, yet declared the same
recovery API, so it installed, crashed at boot and rolled back. Now the app
declares the oldest API it runs on (app.RECOVERY_API), falls back for
anything newer (app/recovery.py), and the recovery layer refuses a bundle
that needs more than it has, before staging anything."""

import json

import pytest

import devsim
from devsim import wait_for
from test_updates import push, running

OLD = "52cda75"  # the recovery layer USB-installed on the tree on 2026-10-18


def status(d):
    return d.get("/")[1]


@pytest.fixture(scope="module")
def old(tmp_path_factory):
    """52cda75's recovery layer and its app (built with today's tool:
    effects/ and tools/ are unchanged since, and it declared API 2)."""
    tmp = tmp_path_factory.mktemp("old")
    fw = devsim.git_tree(OLD, tmp / "src")
    src = (fw / "bootmgr.py").read_text()
    assert "\nAPI = 2 " in src and "def scratch_set" not in src
    bundle = devsim.build_bundle(tmp / "bundle", version="old", generation=1500, app=fw / "app", recovery_api=2)
    return {"recovery": fw, "bundle": bundle}


@pytest.fixture(scope="module")
def needs_newer(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("newer")
    return {api: devsim.build_bundle(tmp / str(api), version=f"api{api}", generation=2500, recovery_api=api)
            for api in (3, 4)}


def manifest(bundle):
    data = bundle.read_bytes()
    n = int.from_bytes(data[8:12], "big")
    return json.loads(data[12:12 + n])


def test_head_bundle_declares_the_oldest_recovery_api_it_runs_on(bundles):
    m = manifest(bundles["v2"])
    assert m["bootmgr"] == 2, m["bootmgr"]
    src = (devsim.ROOT / "firmware" / "bootmgr.py").read_text()
    assert "\nAPI = 3 " in src  # the recovery layer itself is newer


def test_head_app_ota_onto_old_recovery_layer_confirms_healthy(make_device, bundles, old):
    d = make_device(old["bundle"], recovery=old["recovery"])
    running(d, "old", confirmed=True)
    code, body = push(d, bundles["v2"])
    assert code == 200 and body["installed"] == "v2", body
    st = running(d, "v2", confirmed=True, timeout=90)
    bs = d.boot_state()
    assert bs["active"] == "b" and not bs["trial"] and "confirmed b(v2)" in bs["events"], bs
    assert st["recovery"] == {"api": 2, "outdated": True,
                              "missing": ["scratch_set", "scratch_get", "deliberate", "watchdog_reason"],
                              "message": "recovery layer outdated (USB install recommended)"}
    # Reported, but not a health problem.
    _, h = d.get("/health")
    assert h["ok"] and h["healthy"] and h["warnings"] == ["recovery layer outdated (USB install recommended)"], h
    # The post-update reboot was deliberate: the app reads REASON itself.
    assert st["boot"]["reset_cause"] == "reboot", st["boot"]
    assert st["effect"]["quarantined"] == []


def test_resets_are_classified_on_old_recovery_layer(make_device, bundles, old):
    """Best effort until the next USB install: /reboot and Ctrl-C clear the
    markers app-side, and a genuine hang is still quarantined."""
    d = make_device(bundles["v2"], recovery=old["recovery"])
    running(d, "v2", confirmed=True)
    d.post("/state", {"effect": "solid"})
    wait_for(lambda: status(d)["effect"]["running"] == "solid", 10)
    import time

    time.sleep(3)  # saved; scratch 0 names solid
    d.post("/reboot")
    wait_for(lambda: len(d.boots) >= 1, 20, what="reset")
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "reboot" and st["effect"]["quarantined"] == [], st
    d.interrupt()  # mpremote takes over: API 2's main.py doesn't clear the markers, the app does
    wait_for(lambda: len(d.boots) >= 2, 20, what="reset")
    st = d.wait_http(30)
    assert st["effect"]["quarantined"] == [], st["effect"]
    from test_resets import LATE_HANG

    code, _ = d.request("PUT", "/effects/latehang?activate=1", body=LATE_HANG)
    assert code == 200
    wait_for(lambda: len(d.boots) >= 3 and d.boots[2] == "watchdog reset", 40, what="watchdog reset")
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "watchdog"
    assert "latehang" in st["effect"]["quarantined"], st["effect"]


def test_old_recovery_refuses_a_bundle_needing_a_newer_api_before_staging(make_device, old, needs_newer):
    d = make_device(old["bundle"], recovery=old["recovery"])
    running(d, "old", confirmed=True)
    before = d.boot_state()
    code, body = push(d, needs_newer[3])
    assert code == 400 and "newer recovery layer" in body["error"], body
    assert not (d.root / "fw" / "new").exists()
    assert not (d.root / "fw" / "b" / "manifest.json").exists()
    assert d.boot_state()["active"] == before["active"] and status(d)["version"] == "old"


def test_head_recovery_accepts_api3_and_refuses_api4(make_device, bundles, needs_newer):
    d = make_device(bundles["v2"])
    running(d, "v2", confirmed=True)
    code, body = push(d, needs_newer[4])
    assert code == 400 and "newer recovery layer" in body["error"], body
    assert not (d.root / "fw" / "new").exists()
    code, body = push(d, needs_newer[3])
    assert code == 200 and body["installed"] == "api3", body
    st = running(d, "api3", confirmed=True)
    assert st["recovery"]["api"] == 3 and not st["recovery"]["outdated"]
    _, h = d.get("/health")
    assert h["ok"] and h["warnings"] == []
    # Free heap after a collection is what's reported (not the pre-GC sawtooth).
    sup = wait_for(lambda: (lambda s: s if s["heap_live"] else None)(status(d)["supervisor"]), 15)
    assert sup["heap_live_min"] <= sup["heap_live"] and sup["heap_low_uncollected"] <= sup["heap_live"]
