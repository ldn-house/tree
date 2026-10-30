"""Exercise real discovery on MicroPython, optionally with slow serialization.

The cost models a slower MCU without relying on the host's CPU speed. MQTT
accepts immediately: awaiting it must not be mistaken for yielding.
"""
import sys
import os
import time
import json
import asyncio

FW, WORK = sys.argv[1:3]
COST_MS = int(sys.argv[3])
BASELINE = "--baseline" in sys.argv[4:]
HERE = __file__.rsplit("/", 1)[0]
sys.path.insert(0, FW)
sys.path.insert(1, HERE + "/../mocks")
os.chdir(WORK)

from app import fx, ha

os.mkdir("effects")
with open("effects/__init__.py", "w") as f:
    f.write("EFFECTS = ['sample']\n")
sys.path.insert(0, WORK)
os.mkdir("fx")
fx.OVERLAY = WORK + "/fx"
os.mkdir("data")
# Stale bundled metadata for the name overridden on disk at boot.
with open("data/effects.json", "w") as f:
    json.dump({"sample": {"params": {"obsolete": [1, 0, 2]}}}, f)


def overlay(count):
    with open("fx/fx_sample.py", "w") as f:
        f.write("PARAMS = {\n")
        for i in range(count):
            f.write("'p%d': (1, 0, 10),\n" % i)
        f.write("}\n")


overlay(58)  # 29 fixed entities + 58 params = the observed 87 entities.
effects = fx.Effects(WORK)
real_overlay_file = effects.overlay_file
probes = [0]


def probe(name):
    probes[0] += 1
    return real_overlay_file(name)


effects.overlay_file = probe
assert len(effects.meta("sample")["params"]) == 58  # overlay wins at boot
first_probes = probes[0]
for _ in range(10):
    effects.meta("sample")
print("metadata probes for ten cached reads:", probes[0] - first_probes)
if not BASELINE:
    assert probes[0] == first_probes
# Reload metadata after a replacement, then after deletion back to packaged.
overlay(57)
effects.forget("sample")
assert len(effects.meta("sample")["params"]) == 57
overlay(58)
effects.forget("sample")


class State:
    listeners = []
    params = ["param_removed_old"]

    def get(self, key):
        if key == "ha_params":
            return self.params
        return 1

    def param(self, eff, key, default):
        return default

    def set_ha_params(self, ids):
        self.params = ids


class Cfg:
    MQTT_PREFIX = "synthetic_tree"
    HA_PREFIX = "homeassistant"
    HTTP_PORT = 80


class Wifi:
    ip = None


class Updates:
    def self_update_enabled(self):
        return False


class Rt:
    cfg = Cfg()
    state = State()
    fx = effects
    wifi = Wifi()
    updates = Updates()
    n = 500
    version = "synthetic"


class Bus:
    fail_at = None
    sent = []

    async def publish(self, topic, payload, retain=False):
        assert retain
        if self.fail_at == len(self.sent):
            return False
        self.sent.append((topic, payload))
        return True


bus = Bus()
h = ha.HA(Rt(), bus)
real_dumps = json.dumps


def slow_dumps(obj):
    if COST_MS:
        time.sleep_ms(COST_MS)
    return real_dumps(obj)


class SlowJSON:
    dumps = staticmethod(slow_dumps)


# Built-in modules are immutable on MicroPython. Each per-pass discovery
# import gets this proxy, while the test retains the real JSON module.
sys.modules["json"] = SlowJSON()


async def measure(label, force=False):
    done = [False]
    gaps = []

    async def ticker():
        last = time.ticks_ms()
        while not done[0]:
            await asyncio.sleep_ms(0)
            now = time.ticks_ms()
            gaps.append(time.ticks_diff(now, last))
            last = now

    task = asyncio.create_task(ticker())
    await asyncio.sleep_ms(0)
    try:
        result = await h.publish_discovery(force)
    finally:
        done[0] = True
        await task
    longest = max(gaps)
    print("%s: max turn %d ms, %d ticker turns" % (label, longest, len(gaps)))
    # Also covers the skip decision, settings, and a publish that never yields.
    if not BASELINE:
        assert longest < 250, (label, longest)
        assert len(gaps) >= 87, (label, len(gaps))
    assert "app.ha_discovery" not in sys.modules
    import app
    assert not hasattr(app, "ha_discovery")
    return result


async def main():
    global COST_MS
    assert await measure("initial")
    assert len([t for t, p in bus.sent if t.endswith("/config") and p]) == 87
    assert bus.sent[0][0] == h.t_settings
    settings = json.loads(bus.sent[0][1])
    assert len(settings["params"]["sample"]) == 58
    for comp in ("number", "text"):
        assert ("homeassistant/%s/xmas_tree/param_removed_old/config" % comp, b"") in bus.sent
    assert len(h.rt.state.params) == 58
    assert h.discovered is not None and h.discoveries == 1
    bus.sent = []
    assert not await measure("unchanged")
    assert not bus.sent
    assert await measure("forced", True)
    assert bus.sent[0][0] == h.t_settings
    # Timing has been measured for all three decisions. Failure/retry
    # scenarios below need no simulated MCU cost.
    COST_MS = 0
    # Fail settings, a config, and the second stale cleanup. Nothing counts
    # as complete until all sends (including cleanup) succeed.
    for fail_at in (0, 10, 90):
        h.rt.state.params = ["param_removed_old"]
        bus.sent = []
        bus.fail_at = fail_at
        count = h.discoveries
        assert not await h.publish_discovery(True)
        assert h.discovered is None and h.discoveries == count
        bus.sent = []
        bus.fail_at = None
        assert await h.publish_discovery()
        assert h.discovered is not None and h.discoveries == count + 1
    # A replacement must alter the digest and clear the removed param.
    overlay(57)
    effects.forget("sample")
    bus.sent = []
    assert await h.publish_discovery()
    assert ("homeassistant/number/xmas_tree/param_sample_p57/config", b"") in bus.sent
    with open("effects/sample.py", "w") as f:
        f.write("PARAMS = {'packaged': (2, 0, 4)}\n")
    os.remove("fx/fx_sample.py")
    effects.forget("sample")
    assert list(effects.meta("sample")["params"]) == ["packaged"]


asyncio.run(main())
print("discovery behavior verified")
