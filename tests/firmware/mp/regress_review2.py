# Review round 2 reproductions (fw21-review2-*.py), ported to run on the
# MicroPython unix port against any version of firmware/:
#     micropython regress_review2.py <firmware dir> <empty work dir>
# Only calls that exist before and after the fixes are used, so the same
# script shows the failures on the reviewed commit (1d0b869) and their
# absence now. One PASS/FAIL line per finding; exits 1 on any FAIL.
import json
import os
import sys

FW = sys.argv[1]
WORK = sys.argv[2]
sys.path.insert(0, FW)
os.chdir(WORK)

import hashlib

import bootmgr as b

results = []
KEY = "synthetic-key-0123456789"


def check(name, ok, detail):
    results.append(ok)
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def slots(gen=200):
    b.rmtree("fw")
    for s in ("a", "b"):
        b.makedirs("fw/" + s + "/app")
        with open("fw/" + s + "/manifest.json", "w") as f:
            json.dump({"version": s, "generation": gen}, f)


def initial(**kw):
    st = b._default_state()
    st.update(active="a", previous="b")
    st.update(kw)
    b.write_state(st)


def bundle(gen, key=KEY, payload=b"x=1\n"):
    m = {"format": 1, "bootmgr": 1, "version": "synthetic", "generation": gen,
         "files": [{"path": "app/runtime.py", "size": len(payload), "sha256": b.hexlify(hashlib.sha256(payload).digest())}]}
    mb = json.dumps(m).encode()
    sig = b.hmac_sha256(key, mb) if key else b""
    return b.MAGIC + len(mb).to_bytes(4, "big") + mb + len(sig).to_bytes(4, "big") + sig + payload


def try_install(data, key, **kw):
    """True if the bundle verified (it is never committed)."""
    inst = b.Installer(len(data), key, **kw)
    try:
        inst.feed(data)
        for _ in inst.finish():
            pass
        return True, None
    except b.InstallError as e:
        return False, str(e)
    finally:
        inst.abort()


class Mod:
    pass


# R2-H1: a signed config with a null HTTP_PORT replaces both config copies.
for p in ("config.py", b.CONFIG_GOOD):
    try:
        os.remove(p)
    except OSError:
        pass
good = {"WIFI_SSID": "good-net", "HTTP_TOKEN": "good-token-123", "HTTP_PORT": 8080}
with open(b.CONFIG_GOOD, "w") as f:
    json.dump(good, f)
initial()
try:
    b.install_config({"WIFI_SSID": "synthetic-net", "HTTP_TOKEN": "synthetic-token", "HTTP_PORT": None})
    accepted = True
except ValueError:
    accepted = False
with open(b.CONFIG_GOOD) as f:
    kept = json.load(f) == good
check("R2-H1 config with HTTP_PORT null", not accepted and kept,
      "accepted %s, recovery config kept %s" % (accepted, kept))

# R2-H1: a syntactically valid but wrong config must not become the fallback
# just because it was installed (it hasn't reached the network).
try:
    b.install_config({"WIFI_SSID": "wrong-net", "HTTP_TOKEN": "synthetic-token", "HTTP_PORT": 8080})
except ValueError:
    pass
with open(b.CONFIG_GOOD) as f:
    kept = json.load(f) == good
check("R2-H1 unconfirmed config not promoted", kept, "recovery config kept %s" % kept)

# R2-H2: safe-mode blanking with zero timings.
sys.modules.pop("safemode", None)
import safemode

seen = []
fake = Mod()


class Pin:
    OUT = 1

    def __init__(self, *a):
        pass


fake.Pin = Pin
fake.bitstream = lambda pin, enc, timing, data: seen.append(tuple(timing))
safemode.machine = fake
safemode._cfg = lambda k, d=None: {"LED_TIMING_NS": (0, 0, 0, 0), "LED_COUNT": 500}.get(k, d)
safemode._blank()
check("R2-H2 safe-mode blank timing", seen and seen[-1] == (400, 850, 800, 450), "bitstream got %s" % (seen,))

# R2-H3: accepted power settings vs the amp ceiling.
from app.outstage import channel_budget, create

try:
    import cfgcheck

    def accepted(k, v):
        return not cfgcheck.validate({"WIFI_SSID": "x", "HTTP_TOKEN": "t" * 10, k: v})[1]
except ImportError:
    def accepted(k, v):
        return True  # the reviewed version accepted any numeric value

worst = []
for ma, gamma in ((8, None), (8.9, None), (0, None), (-8, None), (8, 0.5), (8, 2.2)):
    if not accepted("LED_MA_PER_CHANNEL", ma) or not accepted("GAMMA", gamma):
        continue
    out = create(500, channel_budget(500, 1.5, ma), gamma, "RGB")
    dst = bytearray(1500)
    out.process(bytes([255]) * 1500, dst, 255)
    amps = sum(dst) * ma / 255000  # under its own (accepted) model
    phys = sum(dst) * 8 / 255000  # and at the physical 8 mA
    if amps > 1.5 + 1e-9 or (ma > 0 and phys > 1.5 * 8 / ma + 1e-9):
        worst.append((ma, gamma, round(amps, 3)))
check("R2-H3 amp ceiling under accepted settings", not worst, "violations %s" % worst)

# R2-M4: lost boot.json with both slots at generation 200.
slots(200)
initial(floor=200)
with open("boot.json", "w") as f:
    f.write("{corrupt")
ok, why = try_install(bundle(100), KEY)
check("R2-M4 floor after lost boot.json", not ok, "generation 100 %s" % ("accepted" if ok else "refused"))
initial(floor="broken")
ok, why = try_install(bundle(100), KEY)
check("R2-M4 floor of the wrong type", not ok, "generation 100 %s" % ("accepted" if ok else "refused"))

# R2-M5: safe-mode low-space reclaim deletes the previous slot first.
slots(200)
initial(floor=200, safe="synthetic recovery")
real_free = b.free_bytes
b.free_bytes = lambda: 0
data = bundle(200)
flags = []
real_installer = b.Installer


class Trap(real_installer):
    def __init__(self, *a, **kw):
        flags.append(kw.get("reclaim_slot"))
        real_installer.__init__(self, *a, **kw)


b.Installer = Trap
try:
    safemode._cfg = lambda k, d=None: {"UPDATE_KEY": KEY}.get(k, d)
    safemode._install(None, {"content-length": str(len(data))}, "", data[:-4], 0)
except Exception:
    pass
b.Installer = real_installer
b.free_bytes = real_free
check("R2-M5 safe-mode upload keeps the previous slot", b.slot_ok("b") and flags and not flags[-1],
      "slot b intact %s, reclaim_slot %s" % (b.slot_ok("b"), flags))

# R2-M6: an empty (interrupted) reserve is repaired.
b.makedirs("fw")
with open(b.RESERVE, "wb") as f:
    pass
b.free_bytes = lambda: 4 * 1024 * 1024
b.ensure_reserve()
b.free_bytes = real_free
size = os.stat(b.RESERVE)[6]
check("R2-M6 partial reserve repaired", size == b.RESERVE_BYTES, "reserve %d of %d bytes" % (size, b.RESERVE_BYTES))

# R2-M8: a signature-bearing older bundle with no key configured.
slots(200)
initial(floor=200)
ok, why = try_install(bundle(100), None)
check("R2-M8 no key, signed older bundle", not ok, "generation 100 %s" % ("accepted" if ok else "refused"))

# R2-L7: downgrade intent by substring.
captured = []


class Capture:
    def __init__(self, *a, **kw):
        captured.append(kw.get("allow_downgrade"))
        raise b.InstallError("captured")


b.Installer = Capture
for q in ("not_allow_downgrade=1", "allow_downgrade=10"):
    safemode._install(None, {"content-length": "100"}, q, b"", 0)
b.Installer = real_installer
check("R2-L7 exact downgrade flag", captured == [False, False], "flags %s" % captured)

sys.exit(0 if all(results) else 1)
