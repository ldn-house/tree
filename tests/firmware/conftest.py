"""Fixtures for the offline firmware tests (MicroPython unix port).

Run:  uv run --with pytest --with paho-mqtt --with mpy-cross==1.27.0.post2 pytest tests/firmware
Needs `micropython` (unix port) and, for the MQTT tests, `mosquitto` on PATH
or at $MOSQUITTO.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import devsim  # noqa: E402

# mp/ holds MicroPython-only tests, run through test_units.py.
collect_ignore = ["mp"]

MOSQUITTO = os.environ.get("MOSQUITTO") or shutil.which("mosquitto") or "/opt/homebrew/opt/mosquitto/sbin/mosquitto"
SIGNING_KEY = "test-signing-key"


def pytest_collection_modifyitems(config, items):
    if not shutil.which(devsim.MICROPYTHON):
        skip = pytest.mark.skip(reason="micropython (unix port) not installed")
        for item in items:
            item.add_marker(skip)


# -- bundles -----------------------------------------------------------------


def _patched_app(tmp: Path, name: str, patch) -> Path:
    d = tmp / f"app-{name}"
    shutil.copytree(devsim.ROOT / "firmware" / "app", d)
    patch(d)
    return d


@pytest.fixture(scope="session")
def bundles(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("bundles")

    gens = {"v1": 1000, "v2": 2000, "v2-signed": 2000, "crash": 3000, "sick": 3000}

    def build(name, **kw):
        return devsim.build_bundle(tmp / name, version=name, generation=gens[name], **kw)

    def crash(d):  # the new firmware dies on import
        p = d / "runtime.py"
        p.write_text(p.read_text() + "\nraise RuntimeError('broken build')\n")

    def sick(d):  # runs, but never reaches trial health (MQTT never connects)
        p = d / "cfg.py"
        p.write_text(p.read_text() + "\nMQTT_BROKER = '127.0.0.1'\nMQTT_PORT = 9\n")

    return {
        "v1": build("v1"),
        "v2": build("v2"),
        "v2-signed": build("v2-signed", key=SIGNING_KEY),
        "crash": build("crash", app=_patched_app(tmp, "crash", crash)),
        "sick": build("sick", app=_patched_app(tmp, "sick", sick)),
    }


# -- devices -----------------------------------------------------------------


@pytest.fixture
def make_device(tmp_path):
    devices = []

    def make(bundle=None, start=True, recovery=None, **config):
        d = devsim.Device(tmp_path / f"dev{len(devices)}", **config)
        d.install_recovery(recovery)
        if bundle is not None:
            d.usb_install(bundle)
        devices.append(d)
        if start:
            d.start()
        return d

    yield make
    for d in devices:
        d.stop()
        if d.hung:
            print(d.console()[-3000:])


@pytest.fixture
def device(make_device, bundles):
    d = make_device(bundles["v1"])
    d.wait_http(40, pred=lambda s: not s["boot"]["trial"])  # USB installs are trials too
    return d


# -- MQTT --------------------------------------------------------------------


class Broker:
    def __init__(self, root: Path):
        self.port = devsim.free_port()
        self.conf = root / "mosquitto.conf"
        self.conf.write_text(f"listener {self.port} 127.0.0.1\nallow_anonymous true\npersistence false\n")
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen([MOSQUITTO, "-c", str(self.conf)], stdout=subprocess.DEVNULL,
                                     stderr=subprocess.DEVNULL)
        devsim.wait_for(self._listening, 10, what="mosquitto")

    def _listening(self):
        import socket

        try:
            socket.create_connection(("127.0.0.1", self.port), timeout=0.5).close()
            return True
        except OSError:
            return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()

    def pause(self):
        self.proc.send_signal(signal.SIGSTOP)

    def resume(self):
        self.proc.send_signal(signal.SIGCONT)


@pytest.fixture
def broker(tmp_path):
    if not Path(MOSQUITTO).exists() and not shutil.which(MOSQUITTO):
        pytest.skip("mosquitto not installed")
    b = Broker(tmp_path)
    b.start()
    yield b
    if b.proc and b.proc.poll() is None:
        b.resume()
    b.stop()


class Bus:
    """Records everything on the broker; also used to publish commands."""

    def __init__(self, port):
        import paho.mqtt.client as mqtt

        self.msgs = []
        self.lock = threading.Lock()
        self.c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.c.on_message = self._on
        self.c.on_connect = lambda c, u, f, rc, p=None: c.subscribe("#")
        self.c.connect("127.0.0.1", port)
        self.c.loop_start()

    def _on(self, c, u, m):
        with self.lock:
            self.msgs.append((m.topic, m.payload, m.retain, time.time()))

    def pub(self, topic, payload, retain=False):
        self.c.publish(topic, payload, retain=retain).wait_for_publish(5)

    def last(self, topic):
        with self.lock:
            for t, p, _r, _ts in reversed(self.msgs):
                if t == topic:
                    return p
        return None

    def last_json(self, topic):
        p = self.last(topic)
        return json.loads(p) if p else None

    def since(self, ts, topic=None):
        with self.lock:
            return [(t, p) for t, p, _r, at in self.msgs if at >= ts and (topic is None or t == topic)]

    def wait(self, topic, pred=lambda p: True, timeout=30, after=0):
        def find():
            with self.lock:
                for t, p, _r, at in reversed(self.msgs):
                    if t == topic and at >= after and pred(p):
                        return p
            return None

        return devsim.wait_for(find, timeout, what=f"MQTT {topic}")

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()


@pytest.fixture
def bus(broker):
    b = Bus(broker.port)
    yield b
    b.close()


def curl(*args):
    """Run curl; returns (http_code, parsed JSON or text). Exit codes other
    than 0 (e.g. 28 when --max-time fires) come back as code 0."""
    r = subprocess.run(["curl", "-s", "-o", "-", "-w", "\n%{http_code}", *args], capture_output=True)
    out = r.stdout.decode(errors="replace")
    body, _, code = out.rpartition("\n")
    try:
        body = json.loads(body) if body else None
    except ValueError:
        pass
    return int(code or 0), body
