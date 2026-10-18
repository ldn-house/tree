"""MQTT and Home Assistant against a real local mosquitto."""

import json
import time

import pytest

from devsim import wait_for

P = "xmas_tree"
AVAIL = P + "/availability"


@pytest.fixture
def mdev(make_device, bundles, broker, bus):
    d = make_device(bundles["v1"], MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    d.wait_http(30)
    bus.wait(AVAIL, lambda p: p == b"online", 30)
    return d


def configs(bus):
    out = {}
    with bus.lock:
        for t, p, _r, _ts in bus.msgs:
            if t.startswith("homeassistant/") and t.endswith("/config"):
                out[t] = json.loads(p) if p else None
    return out


def status(d):
    return d.get("/")[1]


def test_discovery_payloads(mdev, bus):
    wait_for(lambda: len(configs(bus)) >= 20, 20, what="discovery")
    time.sleep(1)
    cfgs = configs(bus)
    uids = [c["unique_id"] for c in cfgs.values()]
    assert len(uids) == len(set(uids)), "unique_ids must be unique"
    for topic, c in cfgs.items():
        assert c["availability_topic"] == AVAIL, topic
        assert c["device"]["identifiers"] == ["xmas_tree_pico"], topic
        assert c["device"]["sw_version"] == "v1"
        assert c["origin"]["name"]
        for k in ("state_topic", "command_topic"):
            if k in c:
                assert c[k].startswith(P + "/"), (topic, k)
    light = cfgs["homeassistant/light/xmas_tree/config"]
    assert light["unique_id"] == "xmas_tree_pico"  # same entity as the old firmware
    assert light["schema"] == "json" and light["supported_color_modes"] == ["brightness"]
    assert light["effect_list"] == ["auto", "solid", "chase", "height"]
    have = {t.split("/")[1] + "." + (t.split("/")[3] if t.count("/") == 4 else "light") for t in cfgs}
    want = {
        "light.light", "number.speed", "number.auto_cycle", "number.param_solid_level", "number.param_chase_speed",
        "sensor.rssi", "sensor.heap", "sensor.uptime", "sensor.fps", "sensor.cap_pct", "sensor.reset",
        "sensor.boot", "sensor.ip", "sensor.error", "sensor.effect_running", "binary_sensor.power_capped",
        "binary_sensor.calibrating", "button.reboot", "button.calibrate_play", "button.calibrate_stop",
        "update.firmware",
    }
    assert want <= have, want - have
    num = cfgs["homeassistant/number/xmas_tree/param_chase_speed/config"]
    assert (num["min"], num["max"], num["step"]) == (0.1, 5.0, 0.1)
    assert num["value_template"] == "{{ value_json.params.chase.speed }}"
    upd = cfgs["homeassistant/update/xmas_tree/firmware/config"]
    assert "command_topic" not in upd  # no UPDATE_KEY, so no install button

    # Published state the templates read from.
    assert bus.last_json(P + "/state") == {"state": "ON", "brightness": 128, "effect": "auto", "color_mode": "brightness"}
    s = bus.wait(P + "/settings")
    assert json.loads(s)["params"]["chase"]["speed"] == 1.0
    diag = json.loads(bus.wait(P + "/diag"))
    for k in ("rssi", "heap", "uptime", "fps", "capped", "cap_pct", "reset", "boot", "ip", "error", "effect"):
        assert k in diag, k
    assert json.loads(bus.wait(P + "/update"))["installed_version"] == "v1"
    assert json.loads(bus.wait(P + "/calibrate/state"))["state"] == "idle"


def test_commands_and_malformed_payloads(mdev, bus):
    d = mdev
    t0 = time.time()
    bus.pub(P + "/set", '{"state":"ON","brightness":40,"effect":"chase"}')
    bus.wait(P + "/state", lambda p: json.loads(p)["brightness"] == 40, 10, after=t0)
    for bad in ("not json", "5", '"ON"', "[1]", '{"state":"maybe"}', '{"brightness":"x"}',
                '{"effect":"nope"}', '{"brightness":null}', "\xff\xfe"):
        bus.pub(P + "/set", bad)
    bus.pub(P + "/cmd/speed", "abc")
    bus.pub(P + "/cmd/speed", "9")  # clamped to 5
    bus.pub(P + "/cmd/auto_cycle", "30")
    bus.pub(P + "/cmd/param/chase/speed", "2.5")
    bus.pub(P + "/cmd/param/chase/bogus", "1")
    bus.pub(P + "/cmd/param/nope/speed", "1")
    bus.pub(P + "/cmd/unknown", "1")
    time.sleep(2)
    st = status(d)["state"]
    assert st["brightness"] == 40 and st["effect"] == "chase" and st["power"]
    assert st["speed"] == 5.0 and st["auto_s"] == 30 and st["params"]["chase"]["speed"] == 2.5
    s = json.loads(bus.last(P + "/settings"))
    assert s["speed"] == 5.0 and s["params"]["chase"]["speed"] == 2.5
    assert "bad command" in d.console()
    bus.pub(P + "/set", '{"state":"OFF"}')
    bus.wait(P + "/state", lambda p: json.loads(p)["state"] == "OFF", 10)
    time.sleep(0.5)
    assert status(d)["effect"]["running"] is None
    assert len(d.boots) == 0


def test_retained_commands_do_not_override_flash(make_device, bundles, broker, bus):
    d = make_device(bundles["v1"], start=False, MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    (d.root / "state.json").write_text(json.dumps({"power": True, "brightness": 50, "effect": "solid"}))
    # Stale retained messages: a retained OFF command and the old firmware's
    # retained state claiming brightness 255.
    bus.pub(P + "/set", '{"state":"OFF"}', retain=True)
    bus.pub(P + "/cmd/speed", "0.2", retain=True)
    bus.pub(P + "/state", '{"state":"ON","brightness":255,"effect":"rainbow_cycle"}', retain=True)
    t0 = time.time()
    d.start()
    bus.wait(AVAIL, lambda p: p == b"online", 30, after=t0)
    time.sleep(2)
    st = status(d)["state"]
    assert st["power"] is True and st["brightness"] == 50 and st["effect"] == "solid" and st["speed"] == 1.0
    assert "ignored retained message" in d.console()
    assert json.loads(bus.last(P + "/state"))["brightness"] == 50  # device state replaces the stale one


def test_last_will_and_broker_restart(mdev, broker, bus):
    d = mdev
    # Broker restart: the device notices, reconnects and republishes everything.
    broker.stop()
    wait_for(lambda: not status(d)["mqtt"]["connected"], 15, what="disconnect noticed")
    time.sleep(3)
    broker.start()
    bus2 = type(bus)(broker.port)
    try:
        t0 = time.time()
        bus2.wait(AVAIL, lambda p: p == b"online", 70)
        took = time.time() - t0
        wait_for(lambda: len(configs(bus2)) >= 20, 20, what="rediscovery")
        assert json.loads(bus2.wait(P + "/state", timeout=10))["state"] == "ON"
        st = status(d)
        assert st["mqtt"]["connects"] >= 2 and len(d.boots) == 0
        print(f"reconnected {took:.1f}s after the broker came back")
        # Power cut: the broker publishes the last will once keepalive lapses.
        t0 = time.time()
        d.kill_now("pwron")
        bus2.wait(AVAIL, lambda p: p == b"offline", 20, after=t0)
        print(f"last will after {time.time() - t0:.1f}s")
        bus2.wait(AVAIL, lambda p: p == b"online", 40, after=time.time())
    finally:
        bus2.close()


def test_black_holed_broker_is_detected(mdev, broker, bus):
    d = mdev
    broker.pause()  # TCP stays up, nothing answers: no RST, no FIN
    try:
        st = wait_for(lambda: (lambda s: s if not s["mqtt"]["connected"] else None)(status(d)), 30,
                      what="dead broker detected")
        assert "no traffic" in st["mqtt"]["last_error"] or st["mqtt"]["last_error"]
        time.sleep(15)  # reconnect attempts time out instead of blocking
        assert len(d.boots) == 0 and status(d)["effect"]["fps"] >= 30
    finally:
        broker.resume()
    wait_for(lambda: status(d)["mqtt"]["connected"], 70, what="reconnect")
    assert len(d.boots) == 0


def test_ha_restart_triggers_rediscovery(mdev, bus):
    time.sleep(2)
    t0 = time.time()
    bus.pub("homeassistant/status", "online")
    bus.wait("homeassistant/light/xmas_tree/config", timeout=15, after=t0)
    bus.wait(P + "/state", timeout=15, after=t0)


def test_reboot_button(mdev, bus):
    d = mdev
    t0 = time.time()
    bus.pub(P + "/cmd/reboot", "PRESS")
    bus.wait(AVAIL, lambda p: p == b"offline", 10, after=t0)
    wait_for(lambda: len(d.boots) == 1, 15)
    bus.wait(AVAIL, lambda p: p == b"online", 30, after=t0 + 0.5)
    assert "requested" in status(d)["boot"]["reset_reason"]


EFF = P + "/effect_running"


def test_running_effect_is_published_on_change(mdev, bus):
    d = mdev
    wait_for(lambda: len(configs(bus)) >= 20, 20, what="discovery")
    sensor = configs(bus)["homeassistant/sensor/xmas_tree/effect_running/config"]
    assert sensor["state_topic"] == EFF and sensor["json_attributes_topic"] == EFF + "/attributes"
    bus.wait(EFF, timeout=10)
    # A manual selection reaches HA within about a second.
    time.sleep(1.2)
    t0 = time.time()
    d.post("/state", {"effect": "chase"})
    bus.wait(EFF, lambda p: p == b"chase", 5, after=t0)
    lag = [at for t, p, _r, at in bus.msgs if t == EFF and p == b"chase" and at >= t0][0] - t0
    assert lag < 1.5, lag
    attrs = json.loads(bus.last(EFF + "/attributes"))
    assert attrs == {"mode": "locked", "selected": "chase", "next_switch_s": None}
    # Auto mode advances by itself (fixture effects run 4 s each) and says so.
    t0 = time.time()
    d.post("/state", {"effect": "auto"})
    names = wait_for(lambda: (lambda ps: ps if len(set(ps)) >= 3 else None)(
        [p for t, p in bus.since(t0, EFF)]), 20, what="auto advances")
    attrs = json.loads(bus.last(EFF + "/attributes"))
    assert attrs["mode"] == "auto" and attrs["selected"] == "auto" and isinstance(attrs["next_switch_s"], int)
    # Power off and calibration.
    t0 = time.time()
    d.post("/state", {"power": False})
    bus.wait(EFF, lambda p: p == b"off", 5, after=t0)
    d.post("/state", {"power": True})
    t0 = time.time()
    d.post("/calibrate/start")
    bus.wait(EFF, lambda p: p == b"calibration", 5, after=t0)
    assert json.loads(bus.last(EFF + "/attributes"))["mode"] == "calibration"
    d.post("/calibrate/stop")
    bus.wait(EFF, lambda p: p not in (b"calibration", b"off"), 5, after=time.time())


def test_running_effect_publishes_are_rate_limited(mdev, bus):
    d = mdev
    bus.wait(EFF, timeout=10)
    time.sleep(1.5)
    t0 = time.time()
    for name in ("solid", "chase", "height", "solid", "chase"):  # flapping
        d.post("/state", {"effect": name})
        time.sleep(0.1)
    time.sleep(3)
    pubs = bus.since(t0, EFF)
    times = [at for t, p, _r, at in bus.msgs if t == EFF and at >= t0]
    assert len(pubs) <= 4, pubs  # ~1/s over ~3.5 s, not one per change
    assert all(b - a >= 0.9 for a, b in zip(times, times[1:])), times
    assert pubs[-1][1] == b"chase"  # the final state still lands
