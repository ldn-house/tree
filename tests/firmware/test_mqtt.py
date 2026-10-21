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
    # "online" goes out before the discovery pass (which first waits up to
    # 1 s for the broker to replay retained state): start tests after it.
    wait_for(lambda: status(d)["ha"]["discoveries"] >= 1, 20, what="first discovery pass")
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
    # Configs arrive in order and the update entity's is the last one.
    wait_for(lambda: "homeassistant/update/xmas_tree/firmware/config" in configs(bus), 20, what="discovery")
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
        "sensor.boot", "sensor.recovery", "sensor.ip", "sensor.error", "sensor.effect_running", "binary_sensor.power_capped",
        "binary_sensor.calibrating", "button.reboot", "button.calibrate_play", "button.calibrate_stop",
        "update.firmware",
    }
    assert want <= have, want - have
    num = cfgs["homeassistant/number/xmas_tree/param_chase_speed/config"]
    assert (num["min"], num["max"], num["step"]) == (0.1, 5.0, 0.1)
    assert num["value_template"] == ("{% set p = value_json.params %}"
                                     "{{ p['chase']['speed'] if 'chase' in p and 'speed' in p['chase'] else 'None' }}")
    # Parsing keys are explicit wherever an earlier firmware's config had
    # them (HA keeps a key a newer config omits).
    eff = cfgs["homeassistant/sensor/xmas_tree/effect_running/config"]
    assert eff["value_template"] == "{{ value }}" and eff["json_attributes_template"] == "{{ value }}"
    light = cfgs["homeassistant/light/xmas_tree/config"]
    assert light["json_attributes_topic"] == P + "/effect_running/attributes"  # not the old xmas_tree/attributes
    assert light["json_attributes_template"] == "{{ value }}"
    upd = cfgs["homeassistant/update/xmas_tree/firmware/config"]
    assert "command_topic" not in upd  # no UPDATE_KEY, so no install button

    # Published state the templates read from.
    assert json.loads(bus.wait(P + "/state", timeout=10)) == {"state": "ON", "brightness": 128, "effect": "auto", "color_mode": "brightness"}
    s = bus.wait(P + "/settings")
    assert json.loads(s)["params"]["chase"]["speed"] == 1.0
    diag = json.loads(bus.wait(P + "/diag"))
    for k in ("rssi", "heap", "uptime", "fps", "capped", "cap_pct", "reset", "boot", "ip", "error", "effect", "recovery"):
        assert k in diag, k
    assert diag["recovery"] == "ok (API 3)"
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
    want = {"speed": 5.0, "auto_s": 30}
    # (params has no "chase" entry until the param command lands.)
    st = wait_for(lambda: (lambda s: s if s["speed"] == 5.0 and s["auto_s"] == 30
                                    and s["params"].get("chase", {}).get("speed") == 2.5 else None)(status(d)["state"]),
                  10, what="commands applied")
    assert st["brightness"] == 40 and st["effect"] == "chase" and st["power"]
    s = json.loads(bus.wait(P + "/settings", lambda p: json.loads(p)["params"]["chase"]["speed"] == 2.5, 10))
    assert s["speed"] == want["speed"]
    assert "bad command" in d.console()
    bus.pub(P + "/set", '{"state":"OFF"}')
    bus.wait(P + "/state", lambda p: json.loads(p)["state"] == "OFF", 10)
    wait_for(lambda: status(d)["effect"]["running"] is None, 5, what="effects stopped")
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
    # Both retained commands (set, cmd/speed) were delivered and ignored...
    wait_for(lambda: d.console().count("ignored retained message") >= 2, 15, what="retained commands ignored")
    # ...and the device's own state replaces the stale retained one.
    bus.wait(P + "/state", lambda p: json.loads(p)["brightness"] == 50, 15, after=t0)
    st = status(d)["state"]
    assert st["power"] is True and st["brightness"] == 50 and st["effect"] == "solid" and st["speed"] == 1.0


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
        time.sleep(15)  # reconnect attempts time out instead of blocking...
        # ...so the LEDs keep their frame rate (40 fps target; one sample can
        # dip on a loaded runner, a blocked loop would not recover).
        wait_for(lambda: status(d)["effect"]["fps"] >= 30, 10, what="frame rate")
        assert len(d.boots) == 0
    finally:
        broker.resume()
    wait_for(lambda: status(d)["mqtt"]["connected"], 70, what="reconnect")
    assert len(d.boots) == 0


def test_ha_restart_triggers_rediscovery(mdev, bus):
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
    wait_for(lambda: "homeassistant/update/xmas_tree/firmware/config" in configs(bus), 20, what="discovery")
    sensor = configs(bus)["homeassistant/sensor/xmas_tree/effect_running/config"]
    assert sensor["state_topic"] == EFF and sensor["json_attributes_topic"] == EFF + "/attributes"
    bus.wait(EFF, timeout=10)
    # A manual selection reaches HA within about a second.
    time.sleep(1.2)
    t0 = time.time()
    d.post("/state", {"effect": "chase"})
    bus.wait(EFF, lambda p: p == b"chase", 5, after=t0)
    lag = [at for t, p, _r, at in bus.msgs if t == EFF and p == b"chase" and at >= t0][0] - t0
    assert lag < 3, lag  # on change (~0.2-1 s here), not on the 30 s diagnostics tick it used to wait for
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
    # The final state lands (possibly already: a publish whose label equals
    # the last one shown is skipped)...
    wait_for(lambda: status(d)["effect"]["running"] == "chase" and bus.last(EFF) == b"chase", 10, what="final state")
    time.sleep(1.5)  # ...and nothing else is published after it
    pubs = bus.since(t0, EFF)
    times = [at for t, p, _r, at in bus.msgs if t == EFF and at >= t0]
    assert len(pubs) <= 4, pubs  # ~1/s over ~3.5 s, not one per change
    assert all(b - a >= 0.9 for a, b in zip(times, times[1:])), times
    assert pubs[-1][1] == b"chase"  # the final state still lands


def test_busy_publisher_keeps_its_session(mdev, bus):
    """Found on the tree: PINGREQ only went out when nothing had been sent
    for keepalive/2, so a tree publishing all the time never pinged, the
    broker never answered, and the session was dropped as dead every
    1.5 x keepalive (29 reconnects in under half an hour). Publish at least
    once a second for 4 x keepalive (5 s): one session, no offline flap."""
    d = mdev
    st = status(d)["mqtt"]
    assert st["connected"] and st["connects"] == 1, st
    t0 = time.time()
    level = 0
    while time.time() - t0 < 20:
        level = 60 if level != 60 else 90
        d.post("/state", {"brightness": level})  # -> a light state publish
        time.sleep(0.5)
    st = status(d)["mqtt"]
    assert st["connected"] and st["connects"] == 1 and st["last_error"] is None, st
    assert st["pings"] >= 3, st  # every keepalive / 2, whatever else was sent
    assert st["tx"] > 30, st
    states = [p for _t, p in bus.since(t0, P + "/state")]
    assert len(states) >= 15, len(states)  # it really was publishing all along
    assert [p for _t, p in bus.since(t0, AVAIL) if p != b"online"] == [], "availability flapped"


def retained(broker, topic, timeout=10):
    """What a fresh subscriber gets for topic: the retained value, or None."""
    import uuid

    import paho.mqtt.client as mqtt

    # After subscribing, publish a sentinel to ourselves: the broker sends a
    # connection's messages in order, so any retained message for topic
    # (queued when the subscription was made) arrives before the sentinel.
    # Its arrival proves "nothing retained" without waiting out a timeout.
    sentinel = "tree-test/sentinel/" + uuid.uuid4().hex
    got, done, failed = [], [], []

    def on_message(c, u, m):
        (done if m.topic == sentinel else got).append(m.payload)

    def on_subscribe(c, u, mid, rcs, p=None):
        if any(rc.is_failure for rc in rcs):
            failed.append(rcs)  # raised below: an assert here would end only paho's thread
        else:
            c.publish(sentinel, b"1")

    c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    c.on_message = on_message
    c.on_connect = lambda c, u, f, rc, p=None: c.subscribe([(topic, 0), (sentinel, 0)])
    c.on_subscribe = on_subscribe
    c.connect("127.0.0.1", broker.port)
    c.loop_start()
    try:
        end = time.time() + timeout
        while not done and not failed and time.time() < end:
            time.sleep(0.02)
    finally:
        c.disconnect()
        c.loop_stop()
    assert not failed, failed
    if not done:
        raise TimeoutError("no answer from the broker about " + topic)
    return got[0] if got else None


def test_availability_survives_an_unclean_drop_and_a_late_will(mdev, broker, bus):
    """On the tree, HA's Running effect went Unavailable and came back stale
    at every reconnect: the broker can publish an old session's will
    ("offline", retained) after the new session's "online", and every
    reconnect re-sent all discovery configs. Now the firmware disconnects
    cleanly when it drops a session, re-asserts "online" after connecting and
    whenever it sees "offline", and re-sends only the states."""
    d = mdev
    assert status(d)["ha"]["discoveries"] == 1
    # An unclean drop: the broker stops answering, the tree gives up on the
    # session (no traffic for 1.5 x 5 s) and reconnects once it's back.
    broker.pause()
    try:
        wait_for(lambda: not status(d)["mqtt"]["connected"], 30, what="drop detected")
    finally:
        broker.resume()
    t_back = time.time()
    st = wait_for(lambda: (lambda s: s if s["mqtt"]["connected"] else None)(status(d)), 70, what="reconnect")
    assert st["mqtt"]["connects"] == 2 and st["mqtt"]["disconnects"] >= 1, st["mqtt"]
    # States are re-published once the new session's discovery decision is
    # made; discovery itself is not (unchanged configs).
    wait_for(lambda: bus.since(t_back, P + "/state") and bus.since(t_back, P + "/effect_running"), 15,
             what="states re-published")
    wait_for(lambda: retained(broker, AVAIL) == b"online", 15, what="retained online")
    assert [t for t, _p in bus.since(t_back) if t.startswith("homeassistant/")] == []
    assert status(d)["ha"]["discoveries"] == 1
    # The running effect still reaches a subscriber after the reconnect.
    t1 = time.time()
    d.post("/state", {"effect": "chase"})
    bus.wait(P + "/effect_running", lambda p: p == b"chase", 5, after=t1)
    # A will from an older session arriving late, over our "online".
    t2 = time.time()
    bus.pub(AVAIL, "offline", retain=True)
    bus.wait(AVAIL, lambda p: p == b"online", 5, after=t2)
    wait_for(lambda: retained(broker, AVAIL) == b"online", 10, what="retained online again")
    wait_for(lambda: status(d)["ha"]["offline_seen"] >= 1, 5, what="offline seen")
    # HA's birth message still forces discovery.
    t3 = time.time()
    bus.pub("homeassistant/status", "online")
    wait_for(lambda: status(d)["ha"]["discoveries"] == 2, 15, what="re-discovery")


def test_dropping_a_session_sends_disconnect_so_no_will_fires(mdev, bus):
    """When the firmware ends a session itself it sends DISCONNECT, so the
    broker discards the will instead of marking the tree offline."""
    d = mdev
    t0 = time.time()
    d.set_net("down")  # the firmware drops the session ("wifi down"); the broker is still there
    st = wait_for(lambda: (lambda s: s if not s["mqtt"]["connected"] else None)(status(d)), 20, what="drop")
    assert st["mqtt"]["disconnects"] == 1, st["mqtt"]
    time.sleep(10)  # well past the broker's 1.5 x keepalive
    assert [p for _t, p in bus.since(t0, AVAIL)] == [], "the will fired"
    d.set_net("up")
    wait_for(lambda: status(d)["mqtt"]["connected"], 40, what="reconnect")
    bus.wait(AVAIL, lambda p: p == b"online", 10, after=t0)


def test_retained_running_effect_tracks_the_latest(mdev, broker, bus):
    """Every effect_running message (state and attributes) is retained, so a
    subscriber that (re)subscribes gets what the tree is showing now, not the
    value from just after connect. (Live subscribers see retain=0: brokers
    clear the flag when forwarding, as MQTT 3.1.1 requires.)"""
    d = mdev
    for eff in ("chase", "height", "solid", "chase"):
        t = time.time()
        d.post("/state", {"effect": eff})
        bus.wait(P + "/effect_running", lambda p, e=eff: p == e.encode(), 5, after=t)
        wait_for(lambda: retained(broker, P + "/effect_running") == eff.encode(), 10, what="retained " + eff)
        wait_for(lambda: (lambda a: a and json.loads(a)["selected"] == eff and json.loads(a)["mode"] == "locked")(
            retained(broker, P + "/effect_running/attributes")), 10, what="retained attributes")
        time.sleep(1.2)  # past the 1 s rate limit before the next change
    # And through auto mode, where the tree switches by itself.
    d.post("/state", {"effect": "auto", "auto_s": 3})
    shown = b"chase"
    for _ in range(3):
        t = time.time()
        shown = bus.wait(P + "/effect_running", lambda p, prev=shown: p != prev, 10, after=t)
        # Retained tracks the latest one (which may already have moved on).
        wait_for(lambda: retained(broker, P + "/effect_running") == bus.last(P + "/effect_running"), 10,
                 what="retained tracks the latest")
        assert json.loads(retained(broker, P + "/effect_running/attributes"))["mode"] == "auto"


def test_first_connect_after_an_update_clears_legacy_state(make_device, bundles, broker, bus):
    """After an OTA the broker still holds the previous firmware's retained
    settings, which lack the new firmware's params, and the old single-file
    firmware's light attributes (xmas_tree/attributes). The discovery pass
    publishes current settings before any param entity's config, so a new
    entity never starts from a payload without its param, and clears the
    legacy topic."""
    bus.pub(P + "/attributes", json.dumps({"ip": "192.0.2.10", "wifi_rssi": -60}), retain=True)
    bus.pub(P + "/settings", json.dumps({"speed": 1.0, "auto_s": 0, "params": {}}), retain=True)
    assert retained(broker, P + "/attributes")
    t0 = time.time()
    d = make_device(bundles["v1"], MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    d.wait_http(30)
    bus.wait(AVAIL, lambda p: p == b"online", 30, after=t0)
    wait_for(lambda: status(d)["ha"]["discoveries"] == 1, 20, what="discovery")
    # The device counts the pass once it has sent the last config; wait for
    # the broker to deliver that one (the update entity's, published last)
    # to this observer too, and with it everything before it, in order.
    last = "homeassistant/update/xmas_tree/firmware/config"
    wait_for(lambda: any(t == last for t, _p in bus.since(t0)), 15, what="the whole discovery pass delivered")
    msgs = bus.since(t0)
    topics = [t for t, _p in msgs]
    assert any("/number/xmas_tree/param_" in t for t in topics), topics
    first_param = min(i for i, t in enumerate(topics) if "/number/xmas_tree/param_" in t)
    settings = [i for i, t in enumerate(topics) if t == P + "/settings"]
    assert settings and settings[0] < first_param, (settings, first_param)
    params = json.loads(msgs[settings[0]][1])["params"]
    for t in topics[first_param:]:
        if "/number/xmas_tree/param_" in t:
            eff, pname = t.split("/")[3][len("param_"):].split("_", 1)
            assert pname in params.get(eff, {}), (t, params)
    assert retained(broker, P + "/attributes") is None  # cleared (empty retained message)
    # Idempotent: a later pass (HA's birth message) clears it again, harmlessly.
    t1 = time.time()
    bus.pub("homeassistant/status", "online")
    wait_for(lambda: status(d)["ha"]["discoveries"] == 2, 20, what="re-discovery")
    # As above: once the pass is delivered, so is its clearing publish (until
    # then a fresh subscriber can receive that empty publish live, as b"").
    wait_for(lambda: any(t == last for t, _p in bus.since(t1)), 15, what="the second pass delivered")
    assert retained(broker, P + "/attributes") is None
