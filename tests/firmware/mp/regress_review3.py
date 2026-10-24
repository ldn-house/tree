# Review round 3 reproductions (fw21-review3-repros.py), ported to run on the
# MicroPython unix port against any version of firmware/:
#     micropython regress_review3.py <firmware dir> <empty work dir>
# with DEVSIM_RESET=wdt DEVSIM_WDT_REASON=1 (a genuine watchdog expiry) in
# the environment. Only calls that exist before and after the fixes are
# used, so the same script shows the failures on the reviewed commit
# (183d9b8) and their absence now. One PASS/FAIL line per finding; exits 1
# on any FAIL.
import json
import os
import sys

FW = sys.argv[1]
WORK = sys.argv[2]
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
sys.path.insert(0, FW)
sys.path.insert(1, HERE + "/../mocks")
os.chdir(WORK)

import hashlib

import bootmgr as b

results = []


def check(name, ok, detail):
    results.append(ok)
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def bundle(gen, payload=b"x=1\n", version="synthetic"):
    m = {"format": 1, "bootmgr": 1, "version": version, "generation": gen,
         "files": [{"path": "app/runtime.py", "size": len(payload), "sha256": b.hexlify(hashlib.sha256(payload).digest())}]}
    mb = json.dumps(m).encode()
    return b.MAGIC + len(mb).to_bytes(4, "big") + mb + (0).to_bytes(4, "big") + payload


def install(data, commit=True):
    inst = b.Installer(len(data), None)
    try:
        inst.feed(data)
        for _ in inst.finish():
            pass
    except b.InstallError as e:
        return str(e)
    if commit:
        b.commit()
    else:
        inst.abort()
    return None


def fresh():
    for p in ("fw", "fx"):
        b.rmtree(p)
    for p in ("boot.json", "config.py", b.CONFIG_PREV, b.CONFIG_GOOD, "info.json"):
        try:
            os.remove(p)
        except OSError:
            pass
    b._config = None
    b.write_state(b._default_state())


def boot():
    """launch()'s state handling for one boot (183d9b8 had it inline)."""
    st = b.read_state()
    kind, _ = b.decide(st)
    if hasattr(b, "config_boot"):
        b.config_boot(st, kind)
    elif st["cfg_trial"]:
        if st["cfg_boots"] >= 1 or kind == "safe":
            b._revert_config(st, "it did not reach the network")
        else:
            st["cfg_boots"] += 1
    b.write_state(st)


def ssid():
    ns = {}
    try:
        with open("config.py") as f:
            exec(f.read(), ns)
    except OSError:
        return None
    return ns.get("WIFI_SSID")


class PowerCut(BaseException):
    pass


# R3-H1: a power cut between replacing config.py and recording the trial
# left the wrong credentials live with no trial to revert them.
stranded = []
real_write = b.write_atomic
for cut in range(3):
    fresh()
    install(bundle(100))
    b.mark_healthy(b.read_state()["active"])
    with open("config.py", "w") as f:
        f.write("WIFI_SSID = 'net'\nHTTP_TOKEN = 'tttttttttt'\n")
    count = [0]

    def write(path, data, cut=cut):
        if count[0] == cut:
            raise PowerCut(path)
        count[0] += 1
        real_write(path, data)

    b.write_atomic = write
    try:
        b.install_config({"WIFI_SSID": "wrong-net", "HTTP_TOKEN": "t" * 10})
    except PowerCut:
        pass
    finally:
        b.write_atomic = real_write
    boot()
    boot()
    if ssid() != "net":
        stranded.append(cut)
check("R3-H1 power cut during a config change", not stranded,
      "wrong credentials still live after two unconfirmed boots when cut before write %s" % stranded if stranded
      else "every cut ends on the old config")

# R3-M3: discovery interrupted part-way was cached as complete.
sys.path.insert(0, FW)
try:
    import asyncio

    from app import ha

    class Cfg:
        MQTT_PREFIX = "xmas_tree"
        HA_PREFIX = "homeassistant"

    class State:
        listeners = []
        params = []

        def get(self, k):
            return list(self.params)

        def set_ha_params(self, ids):
            self.params = list(ids)

    class Rt:
        cfg = Cfg()
        state = State()

    class Bus:
        fail_at = 2
        sent = 0

        async def publish(self, topic, payload, retain=False):
            if self.fail_at is not None and self.sent == self.fail_at:
                return False
            self.sent += 1
            return True

    bus = Bus()
    h = ha.HA(Rt(), bus)
    h.settings_payload = lambda: {}  # later versions publish settings first
    configs = [("homeassistant/sensor/xmas_tree/e%d/config" % i, "{}", "e%d" % i) for i in range(4)]
    if hasattr(h, "_configs"):  # the reviewed firmware kept discovery resident
        h._configs = lambda: iter(configs)
    else:
        from app import ha_discovery
        ha_discovery._configs = lambda _ha: iter(configs)
        h.publish_discovery = lambda force=False: ha_discovery.publish(h, force)
    asyncio.run(h.publish_discovery())
    bus.fail_at = None
    bus.sent = 0
    asyncio.run(h.publish_discovery())  # reconnect, retained availability replayed
    check("R3-M3 interrupted discovery retried", bus.sent >= 4,
          "%d configs sent on reconnect after a drop at the third" % bus.sent)
except Exception as e:
    check("R3-M3 interrupted discovery retried", False, "could not run: %r" % (e,))

# R3-M2: what launch() hands an app after a genuine watchdog expiry. Apps
# written for API 2 (the previous slot after an update, a rollback target)
# quarantine a hanging effect on "wdt" only.
fresh()
RUNTIME = b"""import json
def run(info):
    with open("info.json", "w") as f:
        json.dump({"reset_cause": info.get("reset_cause"), "reset_kind": info.get("reset_kind")}, f)
    raise KeyboardInterrupt
"""
for m in list(sys.modules):
    if m == "app" or m.startswith("app."):
        del sys.modules[m]  # R3-M3 imported the firmware's app; launch() must import the slot's
install(bundle(100, RUNTIME))
b.mark_healthy(b.read_state()["active"])
b.reset = lambda: None
b.launch()
with open("info.json") as f:
    got = json.load(f)
check("R3-M2 launch info reset_cause for API 2 apps", got["reset_cause"] == "wdt",
      "info=%r after a watchdog expiry" % (got,))

# R3-M4: boot.json without a "floor" field read as a stored floor of 0.
fresh()
install(bundle(200, version="v1"))
b.mark_healthy(b.read_state()["active"])
install(bundle(200, version="v2"))
for p in ("boot.json", b.FLOOR_FILE):
    try:
        os.remove(p)
    except OSError:
        pass
with open("boot.json", "w") as f:
    json.dump({"active": "b", "previous": "a", "boots": 0}, f)
floor = b.read_state()["floor"]
replay = install(bundle(100, version="old"), commit=False)
check("R3-M4 missing floor field", floor == 200 and replay is not None,
      "floor %d; generation 100 %s" % (floor, "refused" if replay else "accepted"))

sys.exit(0 if all(results) else 1)
