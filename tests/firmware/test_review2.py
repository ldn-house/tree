"""Device-level regressions for review round 2 (fw21-review-round2.md)."""

import hashlib
import hmac
import json
import time

from conftest import SIGNING_KEY
from devsim import wait_for
from test_updates import running


def status(d):
    return d.get("/")[1]


def put_config(d, values, key=SIGNING_KEY):
    body = json.dumps(values).encode()
    sig = hmac.new(key.encode(), b"tree-config\n" + body, hashlib.sha256).hexdigest()
    return d.request("PUT", "/config", body=body, headers={"X-Signature": sig})


def device_values(d, **changes):
    v = {k: val for k, val in d.config.items() if val is not None}
    v.update(changes)
    return v


def test_r2h1_config_with_wrong_wifi_is_reverted_and_stays_reachable(make_device, bundles):
    d = make_device(bundles["v2-signed"], UPDATE_KEY=SIGNING_KEY, TRIAL_DEADLINE_S=20)
    d.set_net("only:testnet")
    running(d, "v2-signed", confirmed=True)
    wait_for(lambda: (d.root / "config.good.json").exists(), 20, what="config promoted")
    good = json.loads((d.root / "config.good.json").read_text())
    assert good["WIFI_SSID"] == "testnet"
    # Schema rejections: nothing written.
    for bad in (device_values(d, HTTP_PORT=None), device_values(d, LED_MA_PER_CHANNEL=0),
                device_values(d, GAMMA=0.5), device_values(d, LED_TIMING_NS=[0, 0, 0, 0]),
                device_values(d, UPDATE_KEY="a-different-signing-key")):
        code, res = put_config(d, bad)
        assert code == 400, (bad, res)
    assert not (d.root / "config.prev.py").exists()
    # A valid config with an SSID that doesn't exist.
    d.boots.clear()
    code, res = put_config(d, device_values(d, WIFI_SSID="wrong-net"))
    assert code == 200, res
    st = d.wait_http(60, pred=lambda s: s.get("config_trial") and len(d.boots) >= 1)
    # The trial boot can't reach wrong-net, but the last good credentials keep
    # the tree reachable; the trial isn't confirmed through them.
    st = d.wait_http(30, pred=lambda s: s["wifi_fallback"] and s["wifi"]["state"] == "up")
    assert json.loads((d.root / "config.good.json").read_text()) == good
    # At the deadline: reboot, and bootmgr restores the previous config.
    st = d.wait_http(90, pred=lambda s: not s.get("config_trial") and len(d.boots) >= 2)
    assert "testnet" in (d.root / "config.py").read_text()
    assert any("config change reverted" in e for e in d.boot_state()["events"])
    st = d.wait_http(30, pred=lambda s: s["wifi"]["state"] == "up" and not s["wifi_fallback"])
    assert json.loads((d.root / "config.good.json").read_text()) == good


def test_r2h1_good_config_change_is_confirmed_and_promoted(make_device, bundles):
    d = make_device(bundles["v2-signed"], UPDATE_KEY=SIGNING_KEY)
    running(d, "v2-signed", confirmed=True)
    wait_for(lambda: (d.root / "config.good.json").exists(), 20)
    d.boots.clear()
    code, res = put_config(d, device_values(d, HOSTNAME="tree2", POWER_LIMIT_A=1.0))
    assert code == 200, res
    st = d.wait_http(60, pred=lambda s: len(d.boots) >= 1 and not s.get("config_trial"))
    assert st["hostname"] == "tree2" and st["power"]["limit_a"] == 1.0
    good = wait_for(lambda: (lambda g: g if g.get("HOSTNAME") == "tree2" else None)(
        json.loads((d.root / "config.good.json").read_text())), 20, what="promotion")
    assert good["POWER_LIMIT_A"] == 1.0
    assert not (d.root / "config.prev.py").exists()
    assert any("config change confirmed" in e for e in d.boot_state()["events"])


def test_r2h3_invalid_power_settings_never_reach_the_output(make_device, bundles):
    d = make_device(bundles["v1"], LED_MA_PER_CHANNEL=0, GAMMA=0.5)
    st = d.wait_http(30)
    assert "LED_MA_PER_CHANNEL" in st["config_error"] and "GAMMA" in st["config_error"]
    assert st["power"]["ma_per_channel"] == 8  # the rejected values are not used
    d.post("/state", {"effect": "solid", "brightness": 255})
    wait_for(lambda: status(d)["effect"]["running"] == "solid", 10)
    p = wait_for(lambda: (lambda p: p if p["capped"] else None)(status(d)["power"]), 5, what="capped")
    assert p["estimated_a"] <= p["limit_a"] == 1.5, p  # computed from the wire bytes


def test_r2m6_reserve_is_built_after_a_healthy_boot(device):
    st = wait_for(lambda: (lambda s: s if s["reserve"]["complete"] else None)(status(device)), 60,
                  what="reserve built")
    assert st["reserve"]["bytes"] == 256 * 1024
    assert (device.root / "fw" / "floor.json").exists()
