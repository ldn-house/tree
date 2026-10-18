# Review round 1 reproductions (fw21-review-repros.py), ported to run on the
# MicroPython unix port against any version of firmware/bootmgr.py:
#     micropython regress_review1.py <firmware dir> <empty work dir>
# Uses only calls that exist both before and after the fixes, so the same
# script shows the failures on the reviewed commit and their absence now.
# Prints one PASS/FAIL line per finding; exits 1 on any FAIL.
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


def check(name, ok, detail):
    results.append(ok)
    print("%s %s: %s" % ("PASS" if ok else "FAIL", name, detail))


def slots(versions=("1.0.0", "2.0.0")):
    b.rmtree("fw")
    for s, v in zip(("a", "b"), versions):
        b.makedirs("fw/" + s + "/app")
        with open("fw/" + s + "/manifest.json", "w") as f:
            json.dump({"version": v, "generation": 0}, f)


def initial(**kw):
    st = b._default_state()
    st.update(active="a", previous="b")
    st.update(kw)
    b.write_state(st)


def bundle(version, generation, key):
    src = b"x=1\n"
    m = {"format": 1, "bootmgr": 1, "version": version, "generation": generation,
         "files": [{"path": "app/runtime.py", "size": len(src), "sha256": b.hexlify(hashlib.sha256(src).digest())}]}
    mb = json.dumps(m).encode()
    sig = b.hmac_sha256(key, mb)
    return b.MAGIC + len(mb).to_bytes(4, "big") + mb + len(sig).to_bytes(4, "big") + sig + src


def installer(length, key):
    try:
        return b.Installer(length, key, allow_during_trial=True)  # reviewed version's dry-run path
    except TypeError:
        return b.Installer(length, key)


# B1: boot.json can't be written and the app fails at startup.
class Mod:
    pass


slots()
initial()
calls = []
app = Mod()
app.runtime = Mod()


def run(info):
    calls.append("app")
    raise RuntimeError("synthetic startup failure")


app.runtime.run = run
safemode = Mod()
safemode.run = lambda why, info: calls.append("safe")
sys.modules["app"] = app
sys.modules["safemode"] = safemode
real_write, real_reset = b.write_state, b.reset


def enospc(st):
    raise OSError(28, "No space left on device")


b.write_state = enospc
b.reset = lambda: None
b.watchdog = lambda: None
b.stop_feeder = lambda: None
for _ in range(5):
    b.launch()
b.write_state, b.reset = real_write, real_reset
check("B1 unwritable boot state", calls and all(c == "safe" for c in calls), "boots ran: %s" % calls)

# H2: rename fails after the temporary file was written.
slots()
initial(boots=2)
real_os = b.os


class FailingRename:
    def __getattr__(self, n):
        return getattr(real_os, n)

    def rename(self, src, dst):
        raise OSError(28, "injected rename ENOSPC")


b.os = FailingRename()
try:
    b.write_state(dict(b.read_state(), boots=3))
except OSError:
    pass
b.os = real_os
check("H2 rename failure", b.exists("boot.json") and b.read_state()["boots"] == 2,
      "boot.json exists %s, boots %s" % (b.exists("boot.json"), b.read_state()["boots"]))

# H3a: an upload rejected at the header must not touch the previous slot.
slots()
initial()
inst = installer(100, "synthetic-key")
try:
    inst.feed(b"NOTFW!!!" + bytes(92))
except b.InstallError:
    inst.abort()
check("H3 rejected upload", b.slot_ok("b"), "previous slot b intact: %s" % b.slot_ok("b"))

# H3b: dry-run while b is an unconfirmed trial must keep a (its rollback).
slots()
initial(active="b", previous="a", trial=True, boots=1)
data = bundle("probe", 0, "synthetic-key")
inst = installer(len(data), "synthetic-key")
inst.feed(data)
for _ in inst.finish():
    pass
inst.abort()
st = b.read_state()
check("H3 dry-run during trial", b.slot_ok("a") and b.decide(st) == ("app", "a"),
      "rollback target a intact: %s, next boot %s" % (b.slot_ok("a"), b.decide(b.read_state())))

# H5: a correctly signed but older bundle replayed after a newer one was confirmed.
slots()
initial(active="b", previous="a", floor=200)
data = bundle("1.0.0", 100, "synthetic-key")
inst = installer(len(data), "synthetic-key")
try:
    inst.feed(data)
    accepted = True
except b.InstallError as e:
    accepted = False
    reason = str(e)
inst.abort()
check("H5 replayed older bundle", not accepted, "accepted" if accepted else reason)

sys.exit(0 if all(results) else 1)
