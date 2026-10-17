"""Runtime behaviour and fault injection on the simulated device."""

import base64
import time

from conftest import curl
from devsim import wait_for

BOOM = b"""
NAME = "boom"
DURATION_MS = 60000
PARAMS = {}
NEEDS_COORDS = False

class Effect:
    def __init__(self, ctx):
        self.n = ctx.n

    def render(self, t_ms, frame):
        if t_ms > 300:
            raise ValueError("boom at %d" % t_ms)
        for i in range(len(frame)):
            frame[i] = 10
"""

HANG = b"""
NAME = "hang"
DURATION_MS = 60000
PARAMS = {}
NEEDS_COORDS = False

class Effect:
    def __init__(self, ctx):
        pass

    def render(self, t_ms, frame):
        if t_ms > 400:
            while True:
                pass
"""

GOOD = b"""
NAME = "blue"
DURATION_MS = 3000
PARAMS = {"level": (100, 0, 255)}
NEEDS_COORDS = False

class Effect:
    def __init__(self, ctx):
        self.ctx = ctx

    def render(self, t_ms, frame):
        v = int(self.ctx.params["level"])
        for i in range(0, len(frame), 3):
            frame[i] = 0
            frame[i + 1] = 0
            frame[i + 2] = v
"""


def _put(d, name, src, query="?activate=1"):
    return d.request("PUT", f"/effects/{name}{query}", body=src, headers={"Content-Type": "text/x-python"})


def status(d):
    return d.get("/")[1]


def frame_sum(d):
    f = base64.b64decode(d.get("/frame")[1]["frame"])
    return sum(f), f


def test_effects_cycle_and_frame_rate(device):
    d = device
    seen = set()
    end = time.time() + 14
    while time.time() < end:
        st = status(d)
        seen.add(st["effect"]["running"])
        time.sleep(0.5)
    assert {"solid", "chase", "height"} <= seen
    assert st["effect"]["fps"] >= 30, st["effect"]
    assert st["driver"] == "pio+dma"
    assert len(d.boots) == 0


def test_power_cap_applies_before_brightness(device):
    d = device
    budget = status(d)["power_budget"]
    d.post("/state", {"effect": "solid", "brightness": 255})
    wait_for(lambda: status(d)["effect"]["running"] == "solid", 10)
    time.sleep(0.5)
    full, f = frame_sum(d)
    assert budget * 0.97 <= full <= budget, (full, budget)  # capped: as bright as the budget allows
    st = status(d)
    assert st["effect"]["power_capped"] and st["effect"]["cap_factor"] == 12
    d.post("/state", {"brightness": 128})
    time.sleep(0.5)
    half, _ = frame_sum(d)
    assert abs(half - full * 128 / 255) < 1500, (half, full)  # the slider still dims (review H7)
    d.post("/state", {"power": False})
    time.sleep(0.5)
    assert frame_sum(d)[0] == 0


def test_state_persists_across_power_cut(device):
    d = device
    code, body = d.post("/state", {"power": False, "brightness": 77, "effect": "chase", "speed": 2.0})
    assert code == 200 and body["changed"]["brightness"] == 77
    time.sleep(3)  # debounced save
    d.kill_now("pwron")
    st = d.wait_http(30, pred=lambda s: s["uptime_s"] < 20 and len(d.boots) >= 1)
    assert st["state"]["power"] is False and st["state"]["brightness"] == 77
    assert st["state"]["effect"] == "chase" and st["state"]["speed"] == 2.0
    assert st["effect"]["running"] is None  # off, and it stayed off


def test_bad_state_requests(device):
    d = device
    for body in ({"brightness": "x"}, {"power": "ON"}, {"effect": "nope"}, {"params": {"solid": {"zz": 1}}}):
        code, res = d.post("/state", body)
        assert code == 400, (body, res)
    code, res = d.request("POST", "/state", body=b"[1,2", headers={"Content-Type": "application/json"})
    assert code == 400
    code, res = d.post("/state", {"brightness": 999, "params": {"solid": {"level": -5}}})
    assert code == 200 and res["state"]["brightness"] == 255 and res["state"]["params"]["solid"]["level"] == 0


def test_throwing_effect_is_skipped(device):
    d = device
    code, res = _put(d, "boom", BOOM)
    assert code == 200 and res["activated"], res
    st = wait_for(lambda: (lambda s: s if s["effect"]["failed"].get("boom") else None)(status(d)), 20,
                  what="boom marked failed")
    wait_for(lambda: status(d)["effect"]["running"] not in ("boom", None), 20)
    st = status(d)
    assert "boom at" in st["effect"]["failed"]["boom"] and st["effect"]["mode"] == "auto"
    assert st["state"]["effect"] == "boom"  # the user's choice is kept, not overwritten
    assert len(d.boots) == 0
    # Reselecting gives it another chance.
    d.post("/state", {"effect": "solid"})
    d.post("/state", {"effect": "boom"})
    wait_for(lambda: status(d)["effect"]["running"] == "boom", 10)


def test_hanging_effect_is_quarantined_after_watchdog(device):
    d = device
    code, res = _put(d, "hang", HANG, "")
    assert code == 200
    # The user locked the tree to this effect earlier (persisted), so it
    # starts on every boot: without the quarantine this is a reboot loop.
    d.stop()
    import json

    st = json.loads((d.root / "state.json").read_text()) if (d.root / "state.json").exists() else {}
    st["effect"] = "hang"
    (d.root / "state.json").write_text(json.dumps(st))
    d.start()
    wait_for(lambda: len(d.boots) >= 1, 40, what="watchdog reset")
    assert d.boots[0] == "watchdog reset", d.boots
    st = d.wait_http(30)
    assert "hang" in st["effect"]["quarantined"]
    assert st["state"]["effect"] == "hang" and st["effect"]["running"] != "hang"
    # The loop was blocked, so nothing could record a reason; the cause is enough.
    assert st["boot"]["reset_cause"] == "wdt"
    # Healthy (boot counter reset) after 30 s, with no further resets.
    wait_for(lambda: status(d)["supervisor"]["healthy"], 45, what="healthy")
    assert len(d.boots) == 1  # no reboot loop
    assert d.boot_state()["boots"] == 0


def test_hot_push_validation(device):
    d = device
    code, res = _put(d, "broken", b"def render(:\n")
    assert code == 400 and "syntax" in res["error"]
    code, res = _put(d, "noclass", b"X = 1\n")
    assert code == 400 and "Effect" in res["error"]
    code, res = _put(d, "badinit", b"class Effect:\n    def __init__(self, ctx):\n        1/0\n")
    assert code == 400 and "ZeroDivision" in res["error"]
    code, res = _put(d, "Bad-Name", GOOD)
    assert code == 400
    code, res = d.request("PUT", "/effects/blue", body=GOOD, headers={"X-Content-SHA256": "00" * 32})
    assert code == 400 and "SHA256" in res["error"]
    assert "broken" not in status(d)["effects"]

    code, res = _put(d, "blue", GOOD)
    assert code == 200 and res["params"] == {"level": [100, 0, 255]}
    wait_for(lambda: status(d)["effect"]["running"] == "blue", 10)
    time.sleep(0.5)
    f = frame_sum(d)[1]
    assert f[2] > 0 and f[0] == 0 and f[1] == 0
    effects = {e["name"]: e for e in d.get("/effects")[1]["effects"]}
    assert effects["blue"]["overlay"] and effects["blue"]["error"] is None
    # Override a packaged effect, then delete the override.
    code, res = _put(d, "solid", GOOD.replace(b'"blue"', b'"solid"'), "")
    assert code == 200
    assert d.request("DELETE", "/effects/solid")[0] == 200
    assert d.request("DELETE", "/effects/solid")[0] == 404
    assert d.request("DELETE", "/effects/blue")[0] == 200
    st = status(d)
    assert "blue" not in st["effects"] and st["state"]["effect"] == "auto"
    assert len(d.boots) == 0


def test_missing_and_short_coordinates(make_device, bundles):
    d = make_device(bundles["v1"])
    st = d.wait_http(30)
    assert st["coords"] == 500
    d.stop()
    for f in (d.root / "fw" / "a" / "data").glob("coords_compact.txt"):
        f.unlink()
    d.start()
    st = d.wait_http(30, pred=lambda s: s["coords"] == 0)
    effects = {e["name"]: e for e in d.get("/effects")[1]["effects"]}
    assert not effects["height"]["eligible"]  # NEEDS_COORDS effects are skipped, nothing crashes
    seen = set()
    for _ in range(20):
        seen.add(status(d)["effect"]["running"])
        time.sleep(0.5)
    assert "height" not in seen and seen & {"solid", "chase"}
    # A short coordinates file (10 of 500 LEDs) is padded, not an IndexError (review H3).
    short = "\n".join("0.1,%f,0.1" % (i / 10) for i in range(10)).encode()
    code, res = d.request("PUT", "/files/coords.txt", body=short)
    assert code == 200 and res["known"] == 10
    d.post("/state", {"effect": "height"})
    wait_for(lambda: status(d)["effect"]["running"] == "height", 10)
    time.sleep(2)
    assert status(d)["effect"]["failed"] == {}
    code, res = d.request("PUT", "/files/coords.txt", body=b"1,2\nnot,numbers,here\n")
    assert code == 400
    assert len(d.boots) == 0  # no resets after the restart


def test_wifi_down_at_boot(make_device, bundles):
    d = make_device(bundles["v1"], start=False)
    d.set_net("down")
    d.start()
    st = d.wait_http(30)  # (loopback still answers in the simulator)
    time.sleep(20)
    st = status(d)
    assert st["wifi"]["state"] in ("connecting", "backoff") and st["wifi"]["failures"] >= 1
    assert st["effect"]["fps"] >= 30 and st["effect"]["running"]  # LEDs run regardless
    assert st["boot"]["trial"]  # not confirmed without the network...
    assert len(d.boots) == 0  # ...and no watchdog resets while Wi-Fi is down (review H2)
    d.set_net("up")
    st = d.wait_http(70, pred=lambda s: s["wifi"]["state"] == "up")
    assert st["wifi"]["connects"] == 1
    d.wait_http(30, pred=lambda s: not s["boot"]["trial"])
    assert len(d.boots) == 0


def test_wifi_drop_and_bad_password(device):
    d = device
    d.set_net("down")
    wait_for(lambda: status(d)["wifi"]["state"] != "up", 10)
    d.set_net("badauth")
    wait_for(lambda: status(d)["wifi"]["failures"] >= 2, 30)
    d.set_net("up")
    wait_for(lambda: status(d)["wifi"]["state"] == "up", 70)
    assert len(d.boots) == 0


def test_missing_config_still_runs(make_device, bundles):
    d = make_device(bundles["v1"], start=False)
    (d.root / "config.py").write_text("WIFI_SSID = 'x'\nthis is not python\n")
    d.start()
    time.sleep(8)
    assert len(d.boots) == 0
    assert "config.py failed to load" in d.console()
    assert "engine: running" in d.console()


def test_led_timing_and_power_budget_config(make_device, bundles):
    d = make_device(bundles["v1"], LED_TIMING_NS=(350, 900, 750, 500), POWER_BUDGET_MA=3000, LED_MA_PER_CHANNEL=10)
    st = d.wait_http(30)
    assert st["led_timing_ns"] == [350, 900, 750, 500]
    assert st["power"] == {"budget_ma": 3000, "ma_per_channel": 10, "estimated_ma": st["power"]["estimated_ma"],
                           "volts": 12}
    assert st["power_budget"] == 3000 * 255 // 10
    d.post("/state", {"effect": "solid", "brightness": 255})
    wait_for(lambda: status(d)["effect"]["running"] == "solid", 10)
    time.sleep(0.5)
    full, _ = frame_sum(d)
    assert full <= 3000 * 255 // 10 and status(d)["power"]["estimated_ma"] <= 3000
    # A bad timing falls back to the default instead of breaking the driver.
    d2 = make_device(bundles["v1"], LED_TIMING_NS=(400, 850, 800, 100))
    st = d2.wait_http(30)
    assert st["led_timing_ns"] == [400, 850, 800, 450] and st["driver"] == "pio+dma"
    assert "bad LED_TIMING_NS" in d2.console()
