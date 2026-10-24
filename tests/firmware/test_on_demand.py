"""Cold calibration and discovery reloads on the simulated device."""

import json
import shutil
import time

import pytest

import devsim
from devsim import wait_for


@pytest.fixture(scope="module")
def inspected_bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("on-demand")
    app = root / "app"
    shutil.copytree(devsim.ROOT / "firmware" / "app", app)
    p = app / "runtime.py"
    source = p.read_text().replace("        self.calib.routes(self.http)", """        self.calib.routes(self.http)
        self.http.route("GET", "/test-resident", self._test_resident)
""")
    source = source.replace("    # -- shared operations --", """    async def _test_resident(self, req):
        import sys
        import app
        return 200, {
            "static": self.calib.static is not None,
            "bufs": self.calib.bufs is not None,
            "output": self._cal_out is not None,
            "codec": "app.calibseq" in sys.modules,
            "discovery": "app.ha_discovery" in sys.modules or hasattr(app, "ha_discovery"),
        }

    # -- shared operations --""")
    p.write_text(source)
    return devsim.build_bundle(root / "bundle", app=app, version="on-demand", generation=100)


def resident(d):
    return d.get("/test-resident")[1]


@pytest.mark.parametrize("flow", ["http-start", "http-play", "mqtt-play"])
def test_first_calibration_after_boot(flow, inspected_bundle, make_device, broker, bus):
    d = make_device(inspected_bundle, MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port)
    d.wait_http(30, pred=lambda s: not s["boot"]["trial"] and s["ha"]["discoveries"] == 1)
    assert resident(d) == dict(static=False, bufs=False, output=False, codec=False, discovery=False)
    if flow == "http-start":
        assert d.post("/calibrate/start")[0] == 200
        assert d.get("/calibrate/led?n=7", token=True)[0] == 200
        import base64

        want = bytes(21) + bytes([255] * 3) + bytes(1500 - 24)
        wait_for(lambda: base64.b64decode(d.get("/frame")[1]["frame"]) == want, 5, what="first static frame")
        assert resident(d) == dict(static=True, bufs=False, output=True, codec=False, discovery=False)
    else:
        if flow == "http-play":
            assert d.post("/calibrate/play?sequence=default&loops=0&ttl_s=2")[0] == 200
        else:
            bus.pub("xmas_tree/calibrate/set", json.dumps(dict(action="play", sequence="default", loops=0, ttl_s=2)))
        wait_for(lambda: d.get("/calibrate/log")[1]["frames_shown"] >= 2, 5, what="first sequence frames")
        assert resident(d) == dict(static=True, bufs=True, output=True, codec=True, discovery=False)
        wait_for(lambda: d.get("/")[1]["calibration"] == "idle", 6, what="TTL unlock")
        # Reuse the retained buffers, including after the final blank frame.
        assert d.post("/calibrate/play?sequence=default&loops=0&ttl_s=2")[0] == 200
        wait_for(lambda: d.get("/calibrate/log")[1]["frames_shown"] >= 2, 5, what="second sequence")
    d.post("/calibrate/stop")
    wait_for(lambda: d.get("/")[1]["effect"]["running"] is not None, 5, what="effects resumed")
    assert not d.boots and "Traceback" not in d.console()


def test_discovery_reimports_after_birth_and_reconnect(inspected_bundle, make_device, broker, bus):
    d = make_device(inspected_bundle, MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    d.wait_http(30, pred=lambda s: s["ha"]["discoveries"] == 1)
    wait_for(lambda: not resident(d)["discovery"], 5, what="discovery released")
    for expected in (2, 3):
        at = time.time()
        bus.pub("homeassistant/status", "online")
        wait_for(lambda: d.get("/")[1]["ha"]["discoveries"] == expected, 15, what="birth discovery")
        bus.wait("homeassistant/update/xmas_tree/firmware/config", after=at, timeout=10)
        assert not resident(d)["discovery"]
    broker.stop()
    wait_for(lambda: not d.get("/")[1]["mqtt"]["connected"], 15, what="broker disconnect")
    broker.start()  # no retained configs: a full pass is needed
    wait_for(lambda: d.get("/")[1]["ha"]["discoveries"] == 4, 70, what="reconnect discovery")
    assert not resident(d)["discovery"]
    assert not d.boots and "Traceback" not in d.console()
