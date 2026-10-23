"""Takeovers (firmware/app/takeover.py) on the simulated device: TTL expiry,
restore, priority, the user's own changes, calibration, reboots, and the
Home Assistant side over a real local mosquitto."""

import base64
import json
import time

import pytest

from devsim import wait_for

P = "xmas_tree"


def status(d):
    return d.get("/")[1]


def frame(d):
    return base64.b64decode(d.get("/frame")[1]["frame"])


def pixel(d, i=0):
    f = frame(d)
    return tuple(f[3 * i : 3 * i + 3])


def take(d, **cmd):
    return d.post("/takeover", json.dumps(cmd).encode(), headers={"Content-Type": "application/json"})


def state(d, **changes):
    return d.post("/state", json.dumps(changes).encode(), headers={"Content-Type": "application/json"})


def running(d):
    return status(d)["effect"]


def effect_starts(d, name):
    return sum(1 for line in d.get("/log")[1]["lines"] if "engine: running %s " % name in line)


def wait_running(d, name, mode="takeover", timeout=10):
    wait_for(lambda: running(d)["running"] == name and running(d)["mode"] == mode, timeout,
             what="%s running (%s)" % (name, mode))


def test_takeover_shows_over_power_off_and_ttl_restores(device):
    d = device
    assert state(d, power=False, brightness=200)[0] == 200
    wait_for(lambda: sum(frame(d)) == 0, 10, what="dark tree")
    code, body = take(d, id="f1", effect="tint", color="#ff0000", brightness=255, ttl_s=3)
    assert code == 200, body
    assert body["result"] == {"id": "f1", "effect": "tint", "ttl_s": 3, "showing": True}
    wait_running(d, "tint")
    wait_for(lambda: pixel(d)[0] > 0, 10, what="red frame")
    r, g, b = pixel(d)
    assert g == 0 and b == 0
    st = status(d)
    assert st["takeover"]["active"] and st["takeover"]["showing"]["id"] == "f1"
    assert st["takeover"]["showing"]["params"] == {"color": 0xFF0000}
    assert st["state"]["power"] is False  # the user's state is untouched
    # Nobody refreshes it: the dead man's switch ends it and the tree goes
    # back to what it was doing (off).
    wait_for(lambda: not status(d)["takeover"]["active"], 10, what="ttl expiry")
    wait_for(lambda: sum(frame(d)) == 0, 10, what="dark again")
    st = status(d)
    assert st["takeover"]["last_end"] == {"id": "f1", "why": "ttl expired"}
    assert st["effect"]["mode"] != "takeover"
    # Never persisted.
    wait_for(lambda: d.saved_state() is not None and d.saved_state().get("brightness") == 200, 10,
             what="state saved")
    assert "takeover" not in json.dumps(d.saved_state())


def test_refresh_params_priority_and_end(device):
    d = device
    assert take(d, id="f1", effect="tint", color=[0, 0, 255], ttl_s=60)[0] == 200
    wait_running(d, "tint")
    starts = effect_starts(d, "tint")
    # Refreshing with the same command only extends the TTL; new params are
    # applied in place. Neither restarts the effect.
    for _ in range(3):
        assert take(d, id="f1", effect="tint", color=[0, 0, 255], ttl_s=60)[0] == 200
    assert take(d, id="f1", effect="tint", color=[0, 255, 0], ttl_s=60)[0] == 200
    wait_for(lambda: pixel(d)[1] > 0 and pixel(d)[2] == 0, 10, what="green in place")
    assert effect_starts(d, "tint") == starts
    # A refresh only extends the TTL; restart plays the effect from the start.
    code, body = take(d, id="f1", action="refresh", ttl_s=90)
    assert code == 200 and body["result"] == {"id": "f1", "refreshed": True, "ttl_s": 90}
    assert status(d)["takeover"]["showing"]["ttl_s"] > 60
    assert take(d, id="nobody", action="refresh")[1]["result"] == {"id": "nobody", "refreshed": False}
    assert not any(t["id"] == "nobody" for t in status(d)["takeover"]["takeovers"])
    assert effect_starts(d, "tint") == starts
    assert take(d, id="f1", effect="tint", color=[0, 255, 0], ttl_s=60, restart=True)[0] == 200
    wait_for(lambda: effect_starts(d, "tint") == starts + 1, 10, what="restart")
    # A new effect restarts.
    assert take(d, id="f1", effect="chase", ttl_s=60)[0] == 200
    wait_running(d, "chase")
    # A higher priority shows over it, and when it ends the lower returns.
    assert take(d, id="door", effect="tint", color="#ffffff", priority=80, ttl_s=60)[0] == 200
    wait_running(d, "tint")
    assert status(d)["takeover"]["showing"]["id"] == "door"
    # A lower priority one waits underneath.
    assert take(d, id="low", effect="solid", priority=10, ttl_s=60)[0] == 200
    assert status(d)["takeover"]["showing"]["id"] == "door"
    assert d.request("DELETE", "/takeover?id=door")[0] == 200
    wait_running(d, "chase")
    assert take(d, id="f1", action="end")[0] == 200
    wait_running(d, "solid")
    assert d.request("DELETE", "/takeover")[0] == 200
    wait_for(lambda: not status(d)["takeover"]["active"], 10, what="all ended")
    wait_for(lambda: running(d)["mode"] in ("auto", "locked"), 10, what="user's state")


def test_colour_floor_follows_takeover_params(device):
    """The colour floor (effects/_output.py) is found again when a takeover's
    params change in place, as for the user's own params; neither that nor a
    refresh restarts the effect."""
    d = device

    def colour():
        return status(d)["brightness_floor"]["colour"]

    assert take(d, id="f1", effect="solid", params={"tint": 0}, brightness=3, ttl_s=60)[0] == 200
    wait_running(d, "solid")
    wait_for(lambda: colour() == 0, 5, what="white: no colour floor")
    starts = effect_starts(d, "solid")
    assert take(d, id="f1", effect="solid", params={"tint": 1}, brightness=3, ttl_s=60)[0] == 200
    wait_for(lambda: colour() > 10, 5, what="colour floor for warm white, in place")
    warm = colour()
    for _ in range(3):
        assert take(d, id="f1", effect="solid", params={"tint": 1}, brightness=3, ttl_s=60)[0] == 200
    assert colour() == warm
    assert effect_starts(d, "solid") == starts


def test_bad_commands_change_nothing(device):
    d = device
    for bad in ({"effect": "nope"}, {"effect": "tint", "params": {"bogus": 1}}, {"effect": "tint", "color": "red"},
                {"effect": "tint", "ttl_s": 0}, {"effect": "tint", "ttl_s": 99999}, {"effect": "tint", "priority": 101},
                {"effect": "tint", "brightness": "x"}, {"id": "Bad Id!", "effect": "tint"}, {"action": "explode"}):
        code, body = take(d, **bad)
        assert code == 400, (bad, body)
    code, _ = d.post("/takeover", b"not json", headers={"Content-Type": "application/json"})
    assert code == 400
    assert d.post("/takeover", b"{}", token=False)[0] == 401
    assert not status(d)["takeover"]["active"]
    assert status(d)["takeover"]["started"] == 0


def test_user_changes_during_a_takeover(device):
    d = device
    assert state(d, power=True, brightness=255, effect="chase")[0] == 200
    assert take(d, id="f1", effect="tint", color="#ff0000", brightness=255, ttl_s=60)[0] == 200
    wait_running(d, "tint")
    wait_for(lambda: pixel(d)[0] > 0, 10, what="red")
    high = pixel(d)[0]
    # Brightness applies to the takeover showing.
    assert state(d, brightness=60)[0] == 200
    wait_for(lambda: 0 < pixel(d)[0] < high, 10, what="dimmer takeover")
    assert status(d)["takeover"]["showing"]["brightness"] is None
    # Speed and params go to the user's state only.
    assert state(d, speed=2.0)[0] == 200
    assert status(d)["takeover"]["showing"]["speed"] == 1.0
    # Power off ends it and dismisses its id until the user turns the tree on.
    assert state(d, power=False)[0] == 200
    wait_for(lambda: sum(frame(d)) == 0, 10, what="off")
    st = status(d)["takeover"]
    assert not st["active"] and st["dismissed"] == ["f1"]
    code, body = take(d, id="f1", effect="tint", ttl_s=60)
    assert code == 400 and "dismissed" in body["error"]
    assert take(d, id="other", effect="tint", ttl_s=60)[0] == 200  # other ids still work
    assert d.request("DELETE", "/takeover?id=other")[0] == 200
    assert state(d, power=True)[0] == 200
    assert status(d)["takeover"]["dismissed"] == []
    assert take(d, id="f1", effect="tint", ttl_s=60)[0] == 200
    wait_running(d, "tint")
    # Picking an effect ends and dismisses it too; an explicit end lifts that.
    assert state(d, effect="height")[0] == 200
    wait_running(d, "height", mode="locked")
    assert status(d)["takeover"]["dismissed"] == ["f1"]
    assert take(d, id="f1", action="end")[0] == 200
    assert take(d, id="f1", effect="off", ttl_s=60)[0] == 200
    wait_for(lambda: sum(frame(d)) == 0, 10, what="dark takeover")
    assert running(d)["mode"] == "takeover"
    assert status(d)["state"]["power"] is True


def test_calibration_outranks_takeovers(device):
    d = device
    assert take(d, id="f1", effect="tint", color="#00ff00", ttl_s=60)[0] == 200
    wait_running(d, "tint")
    assert d.post("/calibrate/start?ttl_s=60")[0] == 200
    wait_for(lambda: running(d)["mode"] == "calibration", 10, what="calibration")
    # Accepted underneath, TTL running.
    assert take(d, id="f1", effect="tint", color="#0000ff", ttl_s=60)[0] == 200
    assert running(d)["mode"] == "calibration"
    assert d.post("/calibrate/stop")[0] == 200
    wait_running(d, "tint")
    wait_for(lambda: pixel(d)[2] > 0 and pixel(d)[1] == 0, 10, what="blue takeover after calibration")


def test_reboot_drops_takeovers(device):
    d = device
    assert state(d, power=True, effect="height")[0] == 200
    wait_for(lambda: d.saved_state() is not None and d.saved_state().get("effect") == "height", 10, what="saved")
    assert take(d, id="f1", effect="tint", ttl_s=600)[0] == 200
    wait_running(d, "tint")
    assert d.post("/reboot")[0] == 200
    wait_for(lambda: len(d.boots) >= 1, 30, what="reboot")
    st = d.wait_http(30, pred=lambda s: s["uptime_s"] < 30)
    assert not st["takeover"]["active"] and st["takeover"]["started"] == 0
    wait_running(d, "height", mode="locked")


def test_auto_cycle_skips_auto_false(device):
    d = device
    seen = set()
    end = time.time() + 14
    while time.time() < end:
        seen.add(running(d)["running"])
        time.sleep(0.4)
    assert {"solid", "chase", "height"} <= seen and "tint" not in seen
    effects = {e["name"]: e for e in d.get("/effects")[1]["effects"]}
    assert effects["tint"]["auto"] is False and effects["solid"]["auto"] is True
    assert effects["tint"]["params"]["color"] == {"default": 0x102030, "min": 0, "max": 0xFFFFFF,
                                                  "value": 0x102030, "type": "rgb"}
    # Locking to it works as for any effect, and colour params take any form.
    assert state(d, effect="tint", params={"tint": {"color": "#00ff00"}})[0] == 200
    assert d.get("/effects")[1]["effects"][3]["params"]["color"]["value"] == 0x00FF00
    assert state(d, params={"tint": {"color": [1, 2, 3]}})[0] == 200
    assert status(d)["state"]["params"]["tint"]["color"] == 0x010203
    assert state(d, params={"tint": {"color": "blue"}})[0] == 400
    wait_running(d, "tint", mode="locked")


@pytest.fixture
def mdev(make_device, bundles, broker, bus):
    d = make_device(bundles["v1"], MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    d.wait_http(30)
    bus.wait(P + "/availability", lambda p: p == b"online", 30)
    wait_for(lambda: status(d)["ha"]["discoveries"] >= 1, 20, what="first discovery pass")
    return d


def test_home_assistant_takeover(mdev, bus):
    d = mdev
    wait_for(lambda: bus.last("homeassistant/update/xmas_tree/firmware/config") is not None, 20, what="discovery")
    sensor = bus.last_json("homeassistant/binary_sensor/xmas_tree/takeover/config")
    assert sensor["state_topic"] == P + "/takeover/state"
    button = bus.last_json("homeassistant/button/xmas_tree/end_takeover/config")
    assert button["command_topic"] == P + "/cmd/end_takeover"
    text = bus.last_json("homeassistant/text/xmas_tree/param_tint_color/config")
    assert text["pattern"] == "^#[0-9A-Fa-f]{6}$" and text["command_topic"] == P + "/cmd/param/tint/color"
    assert json.loads(bus.wait(P + "/settings", timeout=10))["params"]["tint"]["color"] == "#102030"
    assert state(d, power=False)[0] == 200
    bus.wait(P + "/state", lambda p: json.loads(p)["state"] == "OFF", 10)

    t0 = time.time()
    bus.pub(P + "/takeover/set", json.dumps({"id": "f1", "effect": "tint", "color": [255, 0, 0], "ttl_s": 60}))
    tk = json.loads(bus.wait(P + "/takeover/state", lambda p: json.loads(p)["active"], 10, after=t0))
    assert tk["showing"]["id"] == "f1" and tk["showing"]["effect"] == "tint"
    # The light reports what the tree shows.
    light = json.loads(bus.wait(P + "/state", lambda p: json.loads(p)["state"] == "ON", 10, after=t0))
    assert light["effect"] == "tint" and light["color"] == {"r": 255, "g": 0, "b": 0} and light["color_mode"] == "rgb"
    cfg = bus.last_json("homeassistant/light/xmas_tree/config")
    assert cfg["supported_color_modes"] == ["rgb"] and cfg["brightness"] is True
    attrs = json.loads(bus.wait(P + "/effect_running/attributes", lambda p: json.loads(p)["mode"] == "takeover",
                                10, after=t0))
    assert attrs["takeover"] == "f1"
    assert bus.wait(P + "/effect_running", lambda p: p == b"tint", 10, after=t0)

    # The End takeover button ends and dismisses it; the light goes back to off.
    t1 = time.time()
    bus.pub(P + "/cmd/end_takeover", "PRESS")
    bus.wait(P + "/takeover/state", lambda p: not json.loads(p)["active"], 10, after=t1)
    assert json.loads(bus.wait(P + "/state", lambda p: json.loads(p)["state"] == "OFF", 10, after=t1))
    assert status(d)["takeover"]["dismissed"] == ["f1"]

    # A colour on the light sets the selected effect's colour param.
    t2 = time.time()
    bus.pub(P + "/set", json.dumps({"state": "ON", "effect": "tint"}))
    bus.wait(P + "/state", lambda p: json.loads(p)["effect"] == "tint", 10, after=t2)
    assert status(d)["takeover"]["dismissed"] == []  # turning it on lifts the dismissal
    bus.pub(P + "/set", json.dumps({"state": "ON", "color": {"r": 0, "g": 0, "b": 255}}))
    light = json.loads(bus.wait(P + "/state", lambda p: json.loads(p).get("color", {}).get("b") == 255, 10, after=t2))
    assert light["effect"] == "tint"
    assert status(d)["state"]["params"]["tint"]["color"] == 0x0000FF
    # With an effect named too (a scene restore), the effect wins: the
    # colour only goes to an effect that takes one.
    bus.pub(P + "/set", json.dumps({"state": "ON", "effect": "chase", "color": {"r": 9, "g": 9, "b": 9}}))
    wait_for(lambda: status(d)["state"]["effect"] == "chase", 10, what="effect wins")
    assert "chase" not in status(d)["state"]["params"]
    bus.pub(P + "/set", json.dumps({"state": "ON", "effect": "tint", "color": {"r": 0, "g": 0, "b": 254}}))
    wait_for(lambda: status(d)["state"]["params"]["tint"]["color"] == 0x0000FE, 10, what="tint colour restored")
    # The colour text entity takes #RRGGBB.
    t3 = time.time()
    bus.pub(P + "/cmd/param/tint/color", "#00ff00")
    bus.wait(P + "/settings", lambda p: json.loads(p)["params"]["tint"]["color"] == "#00ff00", 10, after=t3)
    wait_for(lambda: pixel(d)[1] > 0 and pixel(d)[2] == 0, 10, what="green tint")
