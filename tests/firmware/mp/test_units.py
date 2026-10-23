# Unit tests that run on the MicroPython unix port itself:
#     micropython tests/firmware/mp/test_units.py <workdir> [calib.tcal] [calib.sha256]
# (tests/firmware/test_units.py runs this from pytest). Run in an empty
# working directory: the installer writes fw/ and boot.json there.
import gc
import json
import os
import sys

_args = [a if a.startswith("/") else os.getcwd() + "/" + a for a in sys.argv[1:]]

_cwd = os.getcwd()
HERE = __file__.rsplit("/", 1)[0] if "/" in __file__ else "."
if not HERE.startswith("/"):
    HERE = _cwd + "/" + HERE
REPO = HERE + "/../../.."
WORK = _args[0]
sys.path.insert(0, REPO + "/firmware")
sys.path.insert(1, HERE + "/../mocks")
sys.path.insert(2, HERE + "/../fixtures")
os.chdir(WORK)

import hashlib

import bootmgr

failures = []
passed = [0]


def test(fn):
    try:
        fn()
        passed[0] += 1
        print("ok  ", fn.__name__)
    except Exception as e:
        failures.append(fn.__name__)
        print("FAIL", fn.__name__, repr(e))
        sys.print_exception(e)
    return fn


def raises(fn, needle):
    try:
        fn()
    except Exception as e:
        if needle not in str(e):
            raise AssertionError("expected %r in %r" % (needle, str(e)))
        return
    raise AssertionError("expected an error containing %r" % needle)


# -- bundles -----------------------------------------------------------------


def make_bundle(files, key=None, version="t1", mutate=None):
    """files: list of (path, bytes). Mirrors tools/build_firmware.py."""
    manifest = {
        "format": 1,
        "version": version,
        "bootmgr": 2,
        "files": [{"path": p, "size": len(d), "sha256": bootmgr.hexlify(hashlib.sha256(d).digest())} for p, d in files],
    }
    if mutate:
        mutate(manifest)
    mb = json.dumps(manifest).encode()
    sig = bootmgr.hmac_sha256(key, mb) if key else b""
    out = bytearray(bootmgr.MAGIC)
    out += len(mb).to_bytes(4, "big") + mb + len(sig).to_bytes(4, "big") + sig
    for _p, d in files:
        out += d
    return bytes(out)


GOOD = [("app/runtime.py", b"def run(info):\n    pass\n"), ("app/x.py", b"X = 1\n"), ("data/a.txt", b"")]


def reset_flash():
    for p in ("fw", "fx", "boot.json"):
        bootmgr.rmtree(p)


def install(data, key=None, chunk=7, require=None, length=None, **kw):
    inst = bootmgr.Installer(len(data) if length is None else length, key, require_signature=require, **kw)
    try:
        for i in range(0, len(data), chunk):
            inst.feed(data[i : i + chunk])
        for _ in inst.finish():
            pass
    except Exception:
        inst.abort()
        raise
    return inst


def install_commit(data, **kw):
    install(data, **kw)
    return bootmgr.commit(allow_during_trial=True)


@test
def installer_good_bundle_and_commit():
    reset_flash()
    install(make_bundle(GOOD))
    assert bootmgr.exists(bootmgr.STAGE + "/manifest.json") and not bootmgr.exists("fw/a")
    assert bootmgr.commit() == "a"
    assert bootmgr.slot_ok("a") and not bootmgr.exists(bootmgr.STAGE)
    with open("fw/a/app/x.py") as f:
        assert f.read() == "X = 1\n"
    st = bootmgr.read_state()
    assert st["active"] == "a" and st["trial"] and st["boots"] == 0


@test
def installer_truncated_upload_is_rejected():
    reset_flash()
    data = make_bundle(GOOD)
    inst = bootmgr.Installer(len(data), None)
    inst.feed(data[: len(data) - 5])
    raises(lambda: list(inst.finish()), "truncated")
    inst.abort()
    assert not bootmgr.exists("fw/a") and not bootmgr.exists(bootmgr.STAGE)


@test
def installer_length_mismatch_rejected_before_writing():
    reset_flash()
    data = make_bundle(GOOD)
    raises(lambda: install(data, length=len(data) + 100), "Content-Length")
    assert not bootmgr.exists("fw/a") and not bootmgr.exists(bootmgr.STAGE)


@test
def installer_bad_hash():
    reset_flash()
    data = bytearray(make_bundle(GOOD))
    data[-12] ^= 0x01  # inside app/x.py's body (data/a.txt is empty)
    raises(lambda: install(bytes(data)), "sha256 mismatch")
    assert not bootmgr.slot_ok("a") and not bootmgr.exists(bootmgr.STAGE)


@test
def installer_syntax_error_in_py():
    reset_flash()
    files = [("app/runtime.py", b"def run(info:\n")]
    raises(lambda: install(make_bundle(files)), "syntax error")


@test
def installer_rejects_bad_paths():
    for bad in ("../boot.py", "/boot.py", "boot.py", "config.py", "app/../../x.py", "app/X.py", "app/a.exe"):
        reset_flash()
        files = [("app/runtime.py", b"x=1\n"), (bad, b"x")]
        raises(lambda: install(make_bundle(files)), "bad or duplicate path")


@test
def installer_requires_runtime():
    reset_flash()
    raises(lambda: install(make_bundle([("app/other.py", b"x=1\n")])), "app/runtime.py")


@test
def installer_bad_magic_and_trailing_data():
    reset_flash()
    raises(lambda: install(b"NOTFW!!!" + bytes(40)), "not a firmware bundle")
    data = make_bundle(GOOD)
    raises(lambda: install(data + b"x", length=len(data) + 1), "Content-Length")


@test
def installer_signature_policy():
    reset_flash()
    signed = make_bundle(GOOD, key="k1")
    install(signed, key="k1")
    reset_flash()
    raises(lambda: install(signed, key="other"), "bad signature")
    reset_flash()
    raises(lambda: install(make_bundle(GOOD), key="k1"), "not signed")
    reset_flash()
    raises(lambda: install(make_bundle(GOOD), require=True), "UPDATE_KEY")


@test
def installer_newer_bootmgr_rejected():
    reset_flash()

    def m(man):
        man["bootmgr"] = 99

    raises(lambda: install(make_bundle(GOOD, mutate=m)), "newer recovery layer")


@test
def installer_mpy_version_check():
    reset_flash()
    want = sys.implementation._mpy & 0xFF
    ok = bytes([0x4D, want, 0, 31]) + b"\x00" * 8
    install(make_bundle([("app/runtime.py", b"x=1\n"), ("app/y.mpy", ok)]))
    reset_flash()
    old = bytes([0x4D, want - 1, 0, 31]) + b"\x00" * 8
    raises(lambda: install(make_bundle([("app/runtime.py", b"x=1\n"), ("app/y.mpy", old)])), "mpy version")


@test
def installer_targets_inactive_slot_and_refuses_during_trial():
    reset_flash()
    assert install_commit(make_bundle(GOOD)) == "a"
    # Staging during a trial is harmless; replacing the fallback is not.
    inst = install(make_bundle(GOOD, version="t2"))
    assert bootmgr.install_refused() == "the running update is not confirmed yet"
    raises(lambda: bootmgr.commit(), "not confirmed")
    inst.abort()
    bootmgr.mark_healthy("a")
    install(make_bundle(GOOD, version="t2"))
    assert bootmgr.commit() == "b"
    st = bootmgr.read_state()
    assert st["active"] == "b" and st["previous"] == "a"


# -- boot decisions ------------------------------------------------------------


def two_slots(g1=0, g2=0):
    reset_flash()
    install_commit(make_bundle(GOOD, version="v1", mutate=_gen(g1)))
    bootmgr.mark_healthy("a")
    install_commit(make_bundle(GOOD, version="v2", mutate=_gen(g2)))


def _gen(g):
    def m(man):
        if g:
            man["generation"] = g

    return m


@test
def boot_trial_gets_one_boot_then_rolls_back():
    two_slots()
    st = bootmgr.read_state()
    assert bootmgr.decide(st) == ("app", "b")
    st["boots"] += 1
    bootmgr.write_state(st)
    st = bootmgr.read_state()
    assert bootmgr.decide(st) == ("app", "a")
    assert "rolled back" in st["event"] and st["bad"] == "b" and not st["trial"]


@test
def boot_trial_confirmed_stays():
    two_slots()
    st = bootmgr.read_state()
    bootmgr.decide(st)
    st["boots"] += 1
    bootmgr.write_state(st)
    assert bootmgr.mark_healthy("b")
    st = bootmgr.read_state()
    assert bootmgr.decide(st) == ("app", "b") and not st["trial"]


@test
def mark_healthy_only_for_running_slot():
    two_slots()  # b committed, a still running
    assert not bootmgr.mark_healthy("a")
    assert bootmgr.read_state()["trial"]


@test
def boot_repeated_failures_fall_back_then_safe_mode():
    two_slots()
    bootmgr.mark_healthy("b")
    st = bootmgr.read_state()
    st["boots"] = bootmgr.MAX_FAILED_BOOTS
    assert bootmgr.decide(st) == ("app", "a")
    assert "fell back" in st["event"]
    st["boots"] = bootmgr.MAX_FAILED_BOOTS  # a fails too; b is marked bad
    kind, why = bootmgr.decide(st)
    assert kind == "safe" and "boots" in why
    bootmgr.write_state(st)
    bootmgr.leave_safe_mode()
    assert bootmgr.decide(bootmgr.read_state())[0] == "app"


@test
def boot_incomplete_slot_and_empty_flash():
    reset_flash()
    assert bootmgr.decide(bootmgr.read_state())[0] == "safe"
    install_commit(make_bundle(GOOD))
    os.remove("boot.json")
    st = bootmgr.read_state()  # no boot.json: a complete slot is still found
    assert bootmgr.decide(st) == ("app", "a")
    os.remove("fw/a/manifest.json")
    st = {"active": "a", "previous": None, "trial": False, "boots": 0, "bad": None, "safe": None,
          "event": None, "events": []}
    assert bootmgr.decide(st)[0] == "safe"


@test
def boot_json_corrupt_is_recovered():
    reset_flash()
    install_commit(make_bundle(GOOD))
    with open("boot.json", "w") as f:
        f.write("{garbage")
    st = bootmgr.read_state()
    assert bootmgr.decide(st) == ("app", "a")


@test
def manual_rollback():
    two_slots()
    bootmgr.mark_healthy("b")
    assert bootmgr.rollback() == "a"
    st = bootmgr.read_state()
    assert st["active"] == "a" and st["previous"] == "b"


@test
def hmac_matches_rfc4231_case2():
    mac = bootmgr.hmac_sha256(b"Jefe", b"what do ya want for nothing?")
    assert bootmgr.hexlify(mac) == "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"



# -- review round 1 regressions (fw21-review-repros.py and friends) ------------


@test
def b1_unwritable_boot_state_enters_safe_mode_in_ram():
    """Persistent ENOSPC on boot.json plus a failing app: without the fix
    every boot ran the app again with the counter stuck at 0."""

    class Mod:  # no types.ModuleType in MicroPython
        pass

    two_slots()
    bootmgr.mark_healthy("b")
    calls = []
    saved = (bootmgr.write_state, bootmgr.reset, bootmgr.watchdog, bootmgr.stop_feeder, sys.modules.get("app"),
             sys.modules.get("safemode"))

    def enospc(st):
        raise OSError(28, "No space left on device")

    app = Mod()
    app.runtime = Mod()

    def run(info):
        calls.append(("app", info["slot"]))
        raise RuntimeError("synthetic startup failure")

    app.runtime.run = run
    safemode = Mod()
    safemode.run = lambda why, info: calls.append(("safe", why))
    bootmgr.write_state = enospc
    bootmgr.reset = lambda: None
    bootmgr.watchdog = lambda: None
    bootmgr.stop_feeder = lambda: None
    sys.modules["app"] = app
    sys.modules["safemode"] = safemode
    try:
        for _ in range(3):
            bootmgr.launch()
    finally:
        bootmgr.write_state, bootmgr.reset, bootmgr.watchdog, bootmgr.stop_feeder = saved[:4]
        for name, mod in (("app", saved[4]), ("safemode", saved[5])):
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod
    assert calls and all(c[0] == "safe" for c in calls), calls
    assert "cannot record the boot" in calls[0][1]


@test
def b1_reclaim_frees_disposable_space_and_reserve():
    reset_flash()
    install_commit(make_bundle(GOOD))
    bootmgr.mark_healthy("a")
    bootmgr.makedirs("fx")
    with open("fx/fx_big.py", "w") as f:
        f.write("x" * 5000)
    with open("boot.json.tmp", "w") as f:
        f.write("partial")
    bootmgr.makedirs(bootmgr.STAGE)
    with open(bootmgr.RESERVE, "wb") as f:
        f.write(bytes(1000))
    bootmgr.reclaim()
    for p in ("fx", "boot.json.tmp", bootmgr.STAGE, bootmgr.RESERVE):
        assert not bootmgr.exists(p), p
    assert bootmgr.slot_ok("a") and bootmgr.exists("boot.json")


@test
def h2_rename_failure_keeps_the_old_state():
    reset_flash()
    st = bootmgr._default_state()
    st.update(active="a", boots=2)
    bootmgr.write_state(st)
    real = bootmgr.os

    class FailingRename:  # built-in os attributes can't be reassigned
        def __getattr__(self, name):
            return getattr(real, name)

        def rename(self, src, dst):
            raise OSError(28, "injected rename ENOSPC")

    bootmgr.os = FailingRename()
    try:
        raises(lambda: bootmgr.write_state(dict(st, boots=3)), "ENOSPC")
    finally:
        bootmgr.os = real
    assert bootmgr.read_state()["boots"] == 2  # still there, unchanged
    assert not bootmgr.exists("boot.json.tmp")


@test
def h3_rejected_upload_keeps_the_previous_slot():
    two_slots()
    bootmgr.mark_healthy("b")
    for bad in (b"NOTFW!!!" + bytes(92), make_bundle(GOOD, key="other"), make_bundle([("app/runtime.py", b"def (\n")])):
        try:
            install(bad, key="k1" if b"NOTFW" not in bad else None, require=None)
        except bootmgr.InstallError:
            pass
        assert bootmgr.slot_ok("a") and bootmgr.slot_ok("b"), bad[:8]
        assert not bootmgr.exists(bootmgr.STAGE)
    # A hash failure after files were being written: only staging goes.
    data = bytearray(make_bundle(GOOD))
    data[-12] ^= 1
    raises(lambda: install(bytes(data)), "sha256")
    assert bootmgr.slot_ok("a") and bootmgr.slot_ok("b") and not bootmgr.exists(bootmgr.STAGE)


@test
def h3_dry_run_during_trial_keeps_the_rollback_target():
    two_slots()  # b is an unconfirmed trial, a its rollback target
    inst = install(make_bundle(GOOD, version="probe"))
    inst.abort()  # the dry-run path
    assert bootmgr.slot_ok("a") and bootmgr.slot_ok("b")
    st = bootmgr.read_state()
    st["boots"] = 1  # the trial boot failed
    assert bootmgr.decide(st) == ("app", "a")


@test
def h5_older_generation_is_refused_before_anything_is_written():
    two_slots(g1=100, g2=200)
    assert bootmgr.mark_healthy("b")
    assert bootmgr.read_state()["floor"] == 200
    old = make_bundle(GOOD, key="k1", version="old", mutate=_gen(150))
    raises(lambda: install(old, key="k1"), "older than the minimum")
    assert not bootmgr.exists(bootmgr.STAGE) and bootmgr.slot_ok("a") and bootmgr.slot_ok("b")
    raises(lambda: install(make_bundle(GOOD, key="k1"), key="k1"), "older than the minimum")  # no generation
    install(make_bundle(GOOD, key="k1", mutate=_gen(200)), key="k1").abort()  # same generation is fine
    inst = install(old, key="k1", allow_downgrade=True)  # explicit, authenticated downgrade
    assert inst.manifest["version"] == "old"
    inst.abort()


@test
def h5_floor_only_rises_on_confirmation():
    two_slots(g1=100, g2=300)  # 300 installed but never confirmed
    assert bootmgr.read_state()["floor"] == 100
    install(make_bundle(GOOD, mutate=_gen(150))).abort()  # still allowed: 300 never proved itself


def _cfg_reset():
    for p in ("config.py", bootmgr.CONFIG_GOOD, bootmgr.CONFIG_PREV):
        try:
            os.remove(p)
        except OSError:
            pass
    sys.modules.pop("config", None)
    bootmgr._config = None


def _boot():
    """launch()'s state handling for one boot, without running the app."""
    st = bootmgr.read_state()
    kind, _ = bootmgr.decide(st)
    bootmgr.config_boot(st, kind)
    bootmgr.write_state(st)
    return st


class PowerCut(BaseException):
    pass


def _cut_writes_after(n):
    """Make the (n+1)th write_atomic lose power before it lands."""
    real = bootmgr.write_atomic
    count = [0]

    def write(path, data):
        if count[0] == n:
            raise PowerCut(path)
        count[0] += 1
        real(path, data)

    bootmgr.write_atomic = write
    return real


OLDCFG = "WIFI_SSID = 'net'\nWIFI_PASSWORD = 'pw'\nHTTP_TOKEN = 'tttttttttt'\n"
NEWCFG = {"WIFI_SSID": "wrong-net", "WIFI_PASSWORD": "pw", "HTTP_TOKEN": "t" * 10}


def _live_ssid():
    ns = {}
    try:
        with open("config.py") as f:
            exec(f.read(), ns)
    except OSError:
        return None
    return ns.get("WIFI_SSID")


GOODCFG = {"WIFI_SSID": "net", "HTTP_TOKEN": "t" * 10, "HTTP_PORT": 8080}


@test
def m9_config_schema_and_literals():
    _cfg_reset()
    reset_flash()
    bootmgr.write_state(bootmgr._default_state())
    for bad, needle in (({"WIFI_SSID": "x"}, "HTTP_TOKEN is required"),
                        (dict(GOODCFG, HTTP_PORT=None), "HTTP_PORT"),
                        (dict(GOODCFG, HTTP_PORT=70000), "HTTP_PORT"),
                        (dict(GOODCFG, LED_MA_PER_CHANNEL=0), "LED_MA_PER_CHANNEL"),
                        (dict(GOODCFG, GAMMA=0.5), "GAMMA"),
                        (dict(GOODCFG, LED_TIMING_NS=[0, 0, 0, 0]), "LED_TIMING_NS"),
                        (dict(GOODCFG, X=["a"]), "unknown"),
                        (dict(GOODCFG, POWER_LIMIT_A=float("inf")), "POWER_LIMIT_A"),
                        (dict(GOODCFG, MQTT_PORT=True), "MQTT_PORT")):
        raises(lambda: bootmgr.install_config(bad), needle)
    keys = bootmgr.install_config(dict(GOODCFG, WIFI_SSID="net\"'\n", LED_TIMING_NS=[400, 850, 800, 450]))
    assert keys == ["HTTP_PORT", "HTTP_TOKEN", "LED_TIMING_NS", "WIFI_SSID"]
    ns = {}
    with open("config.py") as f:
        exec(f.read(), ns)
    assert ns["WIFI_SSID"] == "net\"'\n" and ns["LED_TIMING_NS"] == (400, 850, 800, 450)
    # Installing doesn't make it the recovery fallback: only reaching the network does.
    assert not bootmgr.exists(bootmgr.CONFIG_GOOD)
    assert bootmgr.read_state()["cfg_trial"]
    raises(lambda: bootmgr.install_config(GOODCFG), "not confirmed")


@test
def r2h1_config_trial_reverts_unless_confirmed():
    _cfg_reset()
    reset_flash()
    install_commit(make_bundle(GOOD))
    bootmgr.mark_healthy("a")
    with open("config.py", "w") as f:
        f.write("WIFI_SSID = 'net'\nHTTP_TOKEN = 'tttttttttt'\n")
    old = open("config.py").read()
    bootmgr.install_config({"WIFI_SSID": "wrong-net", "HTTP_TOKEN": "t" * 10})
    st = bootmgr.read_state()
    assert st["cfg_trial"] and bootmgr.exists(bootmgr.CONFIG_PREV)
    # Boot 1 with the new config: counted, not yet reverted.
    st = bootmgr.read_state()
    kind, _ = bootmgr.decide(st)
    assert kind == "app"
    # launch()'s bookkeeping for two boots without confirmation.
    for _ in range(2):
        _boot()
    assert not bootmgr.read_state()["cfg_trial"]
    assert open("config.py").read() == old  # the previous config is back
    assert "reverted" in bootmgr.read_state()["event"]


@test
def r2h1_promotion_only_after_confirmation():
    if "" not in sys.path:
        sys.path.insert(0, "")  # import config from the work dir, as on the board
    _cfg_reset()
    reset_flash()
    bootmgr.write_state(bootmgr._default_state())
    with open("config.py", "w") as f:
        f.write("WIFI_SSID = 'net'\nHTTP_TOKEN = 'tttttttttt'\nHTTP_PORT = 8080\n")
    values, source, err = bootmgr.load_config()
    assert source == "config.py" and err is None
    assert not bootmgr.exists(bootmgr.CONFIG_GOOD)  # loading alone never promotes
    assert bootmgr.promote_config()
    with open(bootmgr.CONFIG_GOOD) as f:
        assert json.load(f) == values
    # A broken config.py now falls back to that confirmed one.
    with open("config.py", "w") as f:
        f.write("WIFI_SSID = 'x'\nHTTP_PORT = None\n")
    sys.modules.pop("config", None)
    bootmgr._config = None
    values, source, err = bootmgr.load_config()
    assert source == bootmgr.CONFIG_GOOD and "HTTP_PORT" in err and values["HTTP_PORT"] == 8080
    _cfg_reset()


@test
def r2h1_update_key_cannot_change_over_the_network():
    _cfg_reset()
    reset_flash()
    bootmgr.write_state(bootmgr._default_state())
    bootmgr._config = ({"UPDATE_KEY": "k" * 20}, "config.py", None)
    raises(lambda: bootmgr.install_config(dict(GOODCFG)), "UPDATE_KEY")  # removing it
    raises(lambda: bootmgr.install_config(dict(GOODCFG, UPDATE_KEY="z" * 20)), "UPDATE_KEY")
    bootmgr.install_config(dict(GOODCFG, UPDATE_KEY="k" * 20))  # unchanged is fine
    _cfg_reset()


@test
def floor_is_set_after_a_first_install_whose_trial_was_cleared():
    """Bring-up: a first USB install's trial boot couldn't confirm (old Wi-Fi
    credentials) and was cleared with "nothing to roll back to"; the floor
    must still be set once the slot is next marked healthy."""
    reset_flash()
    install_commit(make_bundle(GOOD, mutate=_gen(500)))
    st = bootmgr.read_state()
    st["boots"] = 1  # the trial boot didn't confirm
    bootmgr.decide(st)
    assert "nothing to roll back to" in st["event"] and not st["trial"]
    bootmgr.write_state(st)
    assert bootmgr.read_state()["floor"] == 0
    bootmgr.mark_healthy("a")  # the next boot reaches health (HEALTHY_S)
    assert bootmgr.read_state()["floor"] == 500
    with open(bootmgr.FLOOR_FILE) as f:
        assert json.load(f) == 500


@test
def reset_cause_uses_the_watchdog_reason_register():
    real = bootmgr.machine

    class M:
        PWRON_RESET = 1
        WDT_RESET = 3
        DEVSIM_WATCHDOG_BASE = 0x400D8000

        def __init__(self, cause, reason):
            self.cause = cause
            self.mem32 = {0x400D8000 + 8: reason, 0x400D8000 + 0x0C: 0}

        def reset_cause(self):
            return self.cause

    try:
        for cause, reason, want in ((3, 1, "watchdog"), (3, 2, "reboot"), (1, 0, "pwron"), (3, 3, "watchdog")):
            bootmgr.machine = M(cause, reason)
            assert bootmgr.reset_cause() == want, (cause, reason)
        m = M(3, 2)
        bootmgr.machine = m
        bootmgr.scratch_set(4)
        assert bootmgr.scratch_get() == 4
        bootmgr.deliberate()
        assert bootmgr.scratch_get() is None
    finally:
        bootmgr.machine = real


@test
def r2m5_explicit_reclaim_needs_enough_room():
    two_slots()
    bootmgr.mark_healthy("b")
    real = bootmgr.free_bytes
    bootmgr.free_bytes = lambda: 0
    try:
        data = make_bundle(GOOD)
        raises(lambda: install(data, reclaim_slot=True), "not enough flash even without slot")
        assert bootmgr.slot_ok("a")  # not deleted: it wouldn't have been enough
        raises(lambda: install(data), "not enough flash")
        assert bootmgr.slot_ok("a")
    finally:
        bootmgr.free_bytes = real


@test
def r2m4_floor_copy_survives_boot_json_loss():
    two_slots(g1=100, g2=300)
    bootmgr.mark_healthy("b")
    assert bootmgr.read_state()["floor"] == 300
    with open(bootmgr.FLOOR_FILE) as f:
        assert json.load(f) == 300
    os.remove("boot.json")
    assert bootmgr.read_state()["floor"] == 300  # from the redundant copy
    os.remove(bootmgr.FLOOR_FILE)
    st = bootmgr.read_state()
    assert st["floor"] == 300 and st["floor_source"] == "slots"  # from the slots, not 0


@test
def r2h3_cap_holds_for_every_accepted_setting():
    import cfgcheck
    from app import outstage
    import random

    random.seed(7)
    for ma in (0.5, 1, 7.3, 8, 8.9, 12, 40, 100):
        for limit in (0, 0.2, 1.5, 3, 20):
            for gamma in (None, 1.0, 1.7, 2.2, 4.0):
                clean, err = cfgcheck.validate({"WIFI_SSID": "x", "HTTP_TOKEN": "t" * 10, "LED_MA_PER_CHANNEL": ma,
                                                "POWER_LIMIT_A": limit, "GAMMA": gamma})
                assert not err, err
                out = outstage.create(500, outstage.channel_budget(500, limit, ma), gamma, "RGB")
                dst = bytearray(1500)
                for frame in (bytes([255]) * 1500, bytes(random.getrandbits(8) for _ in range(1500))):
                    out.process(frame, dst, 255)
                    assert outstage.estimated_a(sum(dst), ma) <= limit + 1e-9, (ma, limit, gamma)
    for bad in ({"LED_MA_PER_CHANNEL": 0}, {"LED_MA_PER_CHANNEL": -8}, {"GAMMA": 0.5}, {"POWER_LIMIT_A": -1},
                {"LED_MA_PER_CHANNEL": float("nan")}):
        assert cfgcheck.validate(dict({"WIFI_SSID": "x", "HTTP_TOKEN": "t" * 10}, **bad))[1], bad


@test
def signatures_are_domain_separated():
    body = b"blue\nclass Effect: pass\n"
    sig = bootmgr.hexlify(bootmgr.hmac_sha256("k1", b"tree-effect\n" + body))
    assert bootmgr.signature_ok("effect", body, sig, key="k1")
    assert not bootmgr.signature_ok("config", body, sig, key="k1")
    assert not bootmgr.signature_ok("effect", body + b"x", sig, key="k1")
    assert not bootmgr.signature_ok("effect", body, None, key="k1")
    assert bootmgr.signature_ok("effect", body, None, key="")  # no key configured: token only


@test
def h6_bounded_line_reader():
    import asyncio
    from app.streams import BufReader, TooLong

    class Hostile:
        def __init__(self, data=None):
            self.sent = 0
            self.data = data

        async def read(self, n):
            if self.data is not None:
                out, self.data = self.data[:n], self.data[n:]
                return out
            self.sent += n
            return b"X" * n

    async def main():
        h = Hostile()
        try:
            await BufReader(h, 4096).readline(1024)
        except TooLong:
            pass
        else:
            raise AssertionError("unterminated line accepted")
        assert h.sent <= 1024 + 256, h.sent  # never buffered beyond the limit
        # Many short lines still hit the total header budget.
        many = Hostile(b"a: b\r\n" * 2000)
        br = BufReader(many, 4096)
        try:
            for _ in range(2000):
                await br.readline(1024)
        except TooLong:
            pass
        else:
            raise AssertionError("header budget not enforced")
        ok = BufReader(Hostile(b"GET / HTTP/1.0\r\nHost: x\r\n\r\nBODY"), 4096)
        assert await ok.readline() == b"GET / HTTP/1.0\r\n"
        assert await ok.readline() == b"Host: x\r\n"
        assert await ok.readline() == b"\r\n"
        assert await ok.read(10) == b"BODY"  # read-ahead isn't lost

    asyncio.run(main())


@test
def capped_line_reader_drops_long_lines_within_the_head_budget():
    import asyncio
    from app.streams import BufReader, TooLong

    class Src:
        def __init__(self, data):
            self.data = data
            self.br = None
            self.held = 0

        async def read(self, n):
            self.held = max(self.held, len(self.br.buf))
            out, self.data = self.data[:n], self.data[n:]
            return out

    def reader(data, budget):
        s = Src(data)
        s.br = BufReader(s, budget)
        return s, s.br

    async def main():
        # github.com's redirect: one 9 KB line among short ones.
        csp = b"Content-Security-Policy: " + b"x" * 9200 + b"\r\n"
        s, br = reader(b"HTTP/1.1 302 Found\r\n" + csp + b"Location: /a\r\n\r\nBODY", 16384)
        assert await br.readline_capped(64) == (b"HTTP/1.1 302 Found\r\n", True)
        line, whole = await br.readline_capped(64)
        assert not whole and line == csp[:64], line
        assert await br.readline_capped(64) == (b"Location: /a\r\n", True)
        assert await br.readline_capped(64) == (b"\r\n", True)
        assert await br.read(10) == b"BODY"
        assert s.held <= 64 + 256, s.held  # the long line was never held
        # Dropped bytes still count: an endless line ends at the head budget.
        s, br = reader(b"X-Big: " + b"A" * 100000, 32768)
        try:
            await br.readline_capped(64)
        except TooLong:
            pass
        else:
            raise AssertionError("head budget not enforced")
        assert len(s.data) > 60000 and s.held <= 64 + 256, (len(s.data), s.held)
        # A long last line at EOF.
        s, br = reader(b"y" * 500, 4096)
        assert await br.readline_capped(64) == (b"y" * 64, False)
        assert await br.readline_capped(64) == (b"", True)

    asyncio.run(main())


@test
def l11_render_never_sees_the_clock_limit():
    import asyncio
    from app.engine import Engine, T_RESTART_MS

    for speed in (0.1, 1.0, 5.0):
        seen = []

        class State:
            def get(self, k):
                return speed if k == "speed" else 255

        class Effect:
            def render(self, t, frame):
                seen.append(t)

        class E:
            t_last = 0
            t_frac = 0
            t_ms = T_RESTART_MS - 3
            state = State()
            inst = Effect()
            raw = bytearray(3)
            frames_ok = 0
            name = "synthetic"
            proven = set()
            render_us = 0
            slow = 0
            reload = False
            probing = False
            out = None
            colour = None

            async def present(self, *a):
                pass

            def _tick_fps(self, *a):
                pass

        e = E()
        for now in range(25, 2000, 25):
            if e.reload:
                break
            asyncio.run(Engine._frame(e, now))
        assert e.reload and all(t < T_RESTART_MS for t in seen), (speed, seen[-3:])


# -- output stage --------------------------------------------------------------


@test
def output_cap_applies_before_brightness():
    from app import outstage

    n = 500
    out = outstage.create(n, outstage.channel_budget(n, 1.5, 8), None, "RGB")
    src = bytearray([255] * (3 * n))
    dst = bytearray(3 * n)
    out.process(src, dst, 255)
    assert out.capped and sum(dst) <= out.budget
    full = sum(dst)
    out.process(src, dst, 128)
    half = sum(dst)
    # Brightness still dims when capped: 128 is 4% + 96% x 125/252 above the
    # default floor, and exactly 128/255 without one.
    assert abs(half - full * (4 + 96 * 125 / 252) / 100) <= 3 * n, (full, half)
    out.set_floor(0)
    out.process(src, dst, 128)
    assert abs(sum(dst) - full * 128 // 255) <= 3 * n, (full, sum(dst))
    out.set_floor(outstage.FLOOR_PCT)
    sparse = bytearray(3 * n)
    for i in range(0, 3 * n, 30):
        sparse[i] = 255
    out.process(sparse, dst, 255)
    assert not out.capped and dst == sparse


@test
def output_colour_order():
    from app import outstage

    out = outstage.create(2, 10**6, None, "GRB")
    dst = bytearray(6)
    out.process(bytearray([10, 20, 30, 1, 2, 3]), dst, 255)
    assert list(dst) == [20, 10, 30, 2, 1, 3]


@test
def power_limit_is_in_amps_at_12v():
    from app import outstage

    # Default 1.5 A at 8 mA per channel: channel sum 47812, 12.5% of full white.
    b = outstage.channel_budget(500, 1.5, 8)
    assert b == 47812 and abs(outstage.estimated_a(b, 8) - 1.5) < 0.001
    # A higher measured per-channel draw tightens the cap (same amps).
    assert outstage.channel_budget(500, 1.5, 12) == 31875
    assert outstage.channel_budget(500, 3, 8) == 95625
    # Full white on all 500 pixels: 12 A at 8 mA/channel (~144 W at 12 V).
    assert outstage.estimated_a(500 * 765, 8) == 12.0
    assert outstage.channel_budget(500, 1.5, 8, percent=13) == 49725  # deprecated alias wins


# -- state --------------------------------------------------------------------


@test
def state_validation_and_persistence():
    from app import state

    for p in ("state.json",):
        try:
            os.remove(p)
        except OSError:
            pass
    s = state.State()
    s.load()
    specs = {"rainbow": {"speed": (1.0, 0.1, 5.0)}}
    raises(lambda: s.update({"brightness": "x"}), "not a number")
    raises(lambda: s.update({"power": "ON"}), "power")
    raises(lambda: s.update(5), "object")
    raises(lambda: s.update({"effect": "nope"}, effects=["rainbow"]), "unknown effect")
    raises(lambda: s.update({"params": {"rainbow": {"bogus": 1}}}, param_specs=specs.get), "unknown param")
    ch = s.update({"brightness": 999, "speed": 0, "effect": "rainbow", "params": {"rainbow": {"speed": 50}}},
                  effects=["rainbow"], param_specs=specs.get)
    assert ch["brightness"] == 255 and ch["speed"] == 0.1 and ch["params"]["rainbow"]["speed"] == 5.0
    s.save_now()
    s2 = state.State()
    s2.load()
    assert s2.get("brightness") == 255 and s2.get("effect") == "rainbow"
    with open("state.json", "w") as f:
        f.write("[1,2")
    s3 = state.State()
    s3.load()
    assert s3.get("brightness") == 128


# -- dns / mqtt encoding ---------------------------------------------------------


@test
def dns_parse_answer_with_cname():
    from app import dns

    q = dns._query("github.com", 0x1234)
    # Response: same header with answers: CNAME then A (name pointers to 0x0c).
    resp = bytearray(q)
    resp[2:4] = b"\x81\x80"
    resp[6:8] = b"\x00\x02"
    resp += b"\xc0\x0c\x00\x05\x00\x01\x00\x00\x00\x3c\x00\x02\xc0\x0c"
    resp += b"\xc0\x0c\x00\x01\x00\x01\x00\x00\x00\x3c\x00\x04\x8c\x52\x79\x04"
    assert dns.parse(bytes(resp), 0x1234) == ("140.82.121.4", 60)
    assert dns.is_ip("10.0.0.1") and not dns.is_ip("tree.local")


@test
def mqtt_packet_encoding():
    from app import mqtt

    assert bytes(mqtt._packet(0xC0, b"")) == b"\xc0\x00"
    p = mqtt._packet(0x30, b"x" * 200)
    assert p[0] == 0x30 and p[1] == 0xC8 and p[2] == 0x01
    c = mqtt.MQTT("h", 1, "id", "u", "p", 30, ("t/a", "offline", True))
    pkt = bytes(c._connect_packet())
    assert pkt[0] == 0x10 and b"MQTT\x04" in pkt
    flags = pkt[pkt.index(b"MQTT\x04") + 5]
    assert flags == 0x80 | 0x40 | 0x20 | 0x04 | 0x02


# -- calibration sequences -------------------------------------------------------


def _calib_vector():
    if len(_args) < 3:
        return None, None
    with open(_args[1], "rb") as f:
        data = f.read()
    with open(_args[2]) as f:
        want = f.read().split()[0]
    return data, want


@test
def calib_default_vector_expands_to_reference_hash():
    data, want = _calib_vector()
    if data is None:
        print("     (skipped: no calibration test vector)")
        return
    from app import calib, seqfile

    seq = calib.Sequence(data, 500, 49725)
    h = hashlib.sha256()
    out = bytearray(1500)
    for k in range(seq.frames):
        seq.expand(k, out)
        h.update(out)
    assert bootmgr.hexlify(h.digest()) == want
    # And against the reference expander, frame by frame.
    n, _loops, frames = seqfile.decode(data)
    for k, f in enumerate(frames):
        seq.expand(k, out)
        assert bytes(out) == bytes(f.rgb(n)), k


@test
def calib_colour_frames_match_reference():
    from app import calib, seqfile

    n = 500
    planes = (
        seqfile.pack_plane([i % 3 == 0 for i in range(n)], n),
        seqfile.pack_plane([i % 5 == 0 for i in range(n)], n),
        seqfile.pack_plane([i % 7 == 0 for i in range(n)], n),
    )
    frames = [
        seqfile.Frame(40, (64, 32, 16), planes),
        seqfile.Frame(40, (33, 33, 33), (seqfile.pack_plane([1] * n, n),)),
    ]
    data = seqfile.encode(frames, n, 1)
    seq = calib.Sequence(data, n, 49725)
    out = bytearray(3 * n)
    for k, f in enumerate(frames):
        seq.expand(k, out)
        assert bytes(out) == bytes(f.rgb(n)), k
    assert seq.loop_ms == 80 and seq.over_budget == []


@test
def calib_validation():
    from app import calib, seqfile

    n = 500
    good = seqfile.encode([seqfile.Frame(40, (1, 1, 1), (bytes(63),))], n, 1)
    calib.Sequence(good, n, 1)
    bad = bytearray(good)
    bad[20] ^= 1
    raises(lambda: calib.Sequence(bytes(bad), n, 1), "crc")
    raises(lambda: calib.Sequence(good, 499, 1), "led_count")
    fast = seqfile.encode([seqfile.Frame(5, (1, 1, 1), (bytes(63),))], n, 1)
    raises(lambda: calib.Sequence(fast, n, 1), "min_frame_ms")
    calib.Sequence(fast, n, 1, check_durations=False)
    bright = seqfile.encode([seqfile.Frame(40, (255, 255, 255), (b"\xff" * 63,))], n, 1)
    assert calib.Sequence(bright, n, 49725).over_budget == [0]



@test
def trace_breadcrumb_round_trip():
    import time

    from app import trace

    trace.clear()
    assert trace.take() is None
    trace.armed = 0x80  # as on the board, where the soft timer runs
    trace.at(trace.MQTT)
    trace.tick = time.ticks_add(time.ticks_ms(), -5500)  # the loop stopped turning 5.5 s ago
    trace._check()
    h = trace.take()
    assert h == {"phase": "mqtt", "stalled_s": 5, "timer": True}, h
    assert trace.describe(h) == "hung in mqtt (Python ran 5 s without yielding)"
    assert trace.take() is None  # taken once
    trace.alive(time.ticks_ms())  # the loop turned again: no stall
    trace.at(trace.HTTP)
    assert trace.take() == {"phase": "http request", "stalled_s": 0, "timer": True}
    assert trace.describe({"phase": "http request", "stalled_s": 0, "timer": True}) == \
        "hung in http request (blocked in native code)"
    assert trace.describe({"phase": "gc", "stalled_s": 0, "timer": False}) == "hung in gc"
    trace._check()  # not stalled: nothing recorded
    assert trace.stall == 0
    trace.armed = 0
    trace.clear()


@test
def heap_marks_keep_the_lowest_per_site():
    from app import hw

    hw.heap_sites.clear()
    a = hw.heap_mark("x")
    junk = bytearray(20000)
    b = hw.heap_mark("x")
    assert hw.heap_sites["x"] == min(a, b) and b < a
    del junk
    hw.heap_mark("x")
    assert hw.heap_sites["x"] == b
    hw.heap_sites.clear()


@test
def r3h1_config_change_survives_a_power_cut_at_every_write():
    """Review R3-H1: config.py used to be replaced before boot.json recorded
    the trial, so a power cut between the two left the new (wrong) Wi-Fi
    credentials live with no trial to revert them, and with no good
    snapshot nothing to fall back to. Cut power before each write in turn:
    every outcome is either the old config, or the new one on a trial that
    reverts unless it is confirmed, and two unconfirmed boots always end on
    the old config."""
    for cut in range(4):  # 3 writes: config.prev.py, boot.json, config.py; 3 = no cut
        _cfg_reset()
        reset_flash()
        install_commit(make_bundle(GOOD))
        bootmgr.mark_healthy("a")
        with open("config.py", "w") as f:
            f.write(OLDCFG)
        assert not bootmgr.exists(bootmgr.CONFIG_GOOD)  # the case with nothing else to fall back to
        real = _cut_writes_after(cut)
        try:
            bootmgr.install_config(dict(NEWCFG))
            assert cut == 3, cut
        except PowerCut:
            pass
        finally:
            bootmgr.write_atomic = real
        st = bootmgr.read_state()
        live = _live_ssid()
        assert live == "net" or (live == "wrong-net" and st["cfg_trial"]), (cut, live, st["cfg_trial"])
        st = _boot()
        live = _live_ssid()
        assert live == "net" or (live == "wrong-net" and st["cfg_trial"]), (cut, live, st)
        _boot()  # the new config never reached the network
        assert _live_ssid() == "net", cut
        assert not bootmgr.read_state()["cfg_trial"], cut


@test
def r3h1_interrupted_revert_keeps_the_restored_config():
    """A power cut after the revert put config.prev.py back but before
    boot.json recorded it must not 'revert' again (which, with no
    config.prev.py left, used to delete config.py)."""
    _cfg_reset()
    reset_flash()
    install_commit(make_bundle(GOOD))
    bootmgr.mark_healthy("a")
    with open("config.py", "w") as f:
        f.write(OLDCFG)
    bootmgr.install_config(dict(NEWCFG))
    _boot()  # boot 1: counted
    st = bootmgr.read_state()
    kind, _ = bootmgr.decide(st)
    bootmgr.config_boot(st, kind)  # boot 2 reverts the file... and loses power before write_state
    assert _live_ssid() == "net" and not bootmgr.exists(bootmgr.CONFIG_PREV)
    st = _boot()
    assert _live_ssid() == "net" and not st["cfg_trial"]
    assert "not applied" in st["event"]


@test
def r3h1_safe_mode_config_keeps_the_previous_credentials():
    """From safe mode there is no trial; the replaced config's credentials
    stay a Wi-Fi fallback for safe mode and the app."""
    _cfg_reset()
    reset_flash()
    bootmgr.write_state(bootmgr._default_state())
    with open("config.py", "w") as f:
        f.write(OLDCFG)
    bootmgr.install_config(dict(NEWCFG), trial=False)
    assert _live_ssid() == "wrong-net" and not bootmgr.read_state()["cfg_trial"]
    fb = bootmgr.fallback_configs()
    assert [c["WIFI_SSID"] for c in fb] == ["net"], fb
    # A good snapshot comes first.
    with open(bootmgr.CONFIG_GOOD, "w") as f:
        json.dump({"WIFI_SSID": "good-net", "HTTP_TOKEN": "t" * 10}, f)
    assert [c["WIFI_SSID"] for c in bootmgr.fallback_configs()] == ["good-net", "net"]
    _cfg_reset()


@test
def r3m4_missing_floor_field_uses_the_slots():
    """Review R3-M4: boot.json parsed but without "floor" (and no redundant
    copy) read as a stored floor of 0, so an old signed bundle replayed."""
    two_slots(g1=200, g2=200)
    bootmgr.mark_healthy("b")
    for state in ({}, {"active": "b", "previous": "a", "boots": 0}):
        with open("boot.json", "w") as f:
            json.dump(state, f)
        try:
            os.remove(bootmgr.FLOOR_FILE)
        except OSError:
            pass
        st = bootmgr.read_state()
        assert st["floor"] == 200 and st["floor_source"] == "slots", (state, st["floor"])
        raises(lambda: install(make_bundle(GOOD, version="old", mutate=_gen(100))), "older")
    # A stored floor still counts, and a stored 0 is still a real 0 with no copy.
    with open("boot.json", "w") as f:
        json.dump({"active": "b", "floor": 0}, f)
    assert bootmgr.read_state()["floor"] == 0


@test
def r3m3_interrupted_discovery_is_retried():
    """Review R3-M3: a session dropped part-way through discovery used to be
    recorded as a complete pass (digest cached), so the reconnect skipped
    the configs HA never got."""
    import asyncio

    from app import ha

    class Cfg:
        MQTT_PREFIX = "xmas_tree"
        HA_PREFIX = "homeassistant"

    class State:
        listeners = []

        def __init__(self):
            self.params = ["param_old_gone"]

        def get(self, k):
            return list(self.params)

        def set_ha_params(self, ids):
            self.params = list(ids)

    class Rt:
        cfg = Cfg()
        state = State()

    class Bus:
        def __init__(self, fail_at=None):
            self.sent = []
            self.fail_at = fail_at

        async def publish(self, topic, payload, retain=False):
            if self.fail_at is not None and len(self.sent) == self.fail_at:
                return False  # the session dropped here
            self.sent.append(topic)
            return True

    rt = Rt()
    lead = ["xmas_tree/settings", "xmas_tree/attributes"]  # current settings, then the legacy topic cleared
    bus = Bus(fail_at=len(lead) + 2)
    h = ha.HA(rt, bus)
    h.settings_payload = lambda: {}
    configs = [("homeassistant/sensor/xmas_tree/e%d/config" % i, "{}", "param_e%d" % i) for i in range(4)]
    h._configs = lambda: iter(configs)
    assert asyncio.run(h.publish_discovery()) is False  # interrupted on the third config
    assert bus.sent == lead + [t for t, _j, _o in configs[:2]] and h.discovered is None and h.discoveries == 0
    assert rt.state.params == ["param_old_gone"]  # cleanup record untouched
    # Reconnect, with the broker's retained availability replayed (force=False):
    bus.fail_at = None
    bus.sent = []
    assert asyncio.run(h.publish_discovery()) is True
    assert bus.sent[:2] == lead
    assert bus.sent[2:6] == [t for t, _j, _o in configs]
    assert bus.sent[6:] == ["homeassistant/number/xmas_tree/param_old_gone/config"]  # stale param removed
    assert h.discoveries == 1 and rt.state.params == ["param_e%d" % i for i in range(4)]
    # Complete and unchanged: nothing more to send.
    bus.sent = []
    assert asyncio.run(h.publish_discovery()) is False and bus.sent == []
    # Interrupted during the cleanup of a removed param: retried too.
    rt.state.params = ["param_e0", "param_e1", "param_e2", "param_e3", "param_gone2"]
    configs.pop()
    bus.fail_at = len(lead) + 3
    bus.sent = []
    assert asyncio.run(h.publish_discovery()) is False  # dropped at the removal publish
    assert rt.state.params[-1] == "param_gone2"
    bus.fail_at = None
    bus.sent = []
    assert asyncio.run(h.publish_discovery()) is True
    assert "homeassistant/number/xmas_tree/param_gone2/config" in bus.sent


@test
def reserve_build_waits_out_calibration_playback():
    """Each reserve step is a 4 KB flash write; on the Pico an erase stops
    all code for tens of ms, so a build already running pauses while
    calibration plays (one that hasn't started isn't begun then)."""
    import asyncio

    from app import supervisor

    class Cal:
        playing_now = False

        def playing(self):
            return self.playing_now

    class Rt:
        calib = Cal()

    sup = supervisor.Supervisor(Rt())
    steps = []

    def fake_steps():
        for i in range(4):
            steps.append(i)
            if i == 0:
                Rt.calib.playing_now = True  # playback starts after the first block
            yield

    real = bootmgr.reserve_steps
    bootmgr.reserve_steps = fake_steps
    try:
        async def main():
            t = asyncio.create_task(sup._reserve())
            await asyncio.sleep_ms(700)
            assert steps == [0], steps  # paused for the whole playback
            Rt.calib.playing_now = False
            await asyncio.wait_for(t, 3)
            assert steps == [0, 1, 2, 3], steps

        asyncio.run(main())
    finally:
        bootmgr.reserve_steps = real


@test
def startup_periods_keep_their_device_defaults():
    """The device tests shorten these through config (TRIAL_HEALTHY_S) or a
    patched bundle (conftest "quick"); the firmware's own values stay."""
    from app import supervisor

    class Rt:
        calib = None

    assert supervisor.HEALTHY_S == 30
    assert supervisor.RESERVE_EVERY_S == 600
    sup = supervisor.Supervisor(Rt())
    assert sup._reserve_at + supervisor.RESERVE_EVERY_S == 30  # first reserve check 30 s after boot
    src = open(bootmgr.__file__.rsplit("/", 1)[0] + "/app/cfg.py").read()
    assert 'TRIAL_HEALTHY_S = get("TRIAL_HEALTHY_S", 60)' in src and 'TRIAL_DEADLINE_S = get("TRIAL_DEADLINE_S", 300)' in src


@test
def trial_confirmation_waits_for_task_health():
    """A trial is confirmed only once the supervisor's task checks pass.
    They start after STARTUP_GRACE_MS, so this uses a TRIAL_HEALTHY_S past
    it (the device tests confirm at 1 s, inside the grace, as a shortcut;
    this is the coverage). Runs the real Supervisor.run() on a fake clock."""
    import asyncio

    from app import supervisor

    assert supervisor.STARTUP_GRACE_MS < 6000

    class Stop(Exception):
        pass

    def run(failing):
        clock = [0]
        marked, promoted = [], []

        class T:
            @staticmethod
            def ticks_ms():
                return clock[0]

            @staticmethod
            def ticks_diff(a, b):
                return a - b

        class A:
            @staticmethod
            async def sleep_ms(ms):
                clock[0] += ms
                if clock[0] > 9000:
                    raise Stop

        class B:
            RESERVE_BYTES = 0
            watchdog = staticmethod(lambda: None)
            reserve_size = staticmethod(lambda: 0)
            mark_healthy = staticmethod(lambda slot: marked.append(clock[0]) or True)
            promote_config = staticmethod(lambda: promoted.append(clock[0]) or True)

        class NS:
            def __init__(self, **kw):
                for k, v in kw.items():
                    setattr(self, k, v)

        rt = NS(cfg=NS(WIFI_SSID="testnet", TRIAL_HEALTHY_S=6, TRIAL_DEADLINE_S=25),
                wifi=NS(up=True, using_fallback=False), http=NS(alive=lambda: True), mqtt=None,
                calib=NS(playing=lambda: False), updates=NS(busy=False), state=NS(save_now=lambda: None),
                info={"trial": True, "cfg_trial": True}, slot="a", version="v", ha=None, uptime_ms=0, reboot_at=None)
        saved = supervisor.time, supervisor.asyncio, supervisor.bootmgr, supervisor.write_reason
        supervisor.time, supervisor.asyncio, supervisor.bootmgr = T, A, B
        supervisor.write_reason = lambda why: None
        try:
            sup = supervisor.Supervisor(rt)
            sup.watch("tasks", lambda: "calib task exited" if failing else None)
            try:
                asyncio.run(sup.run())
            except Stop:
                pass
        finally:
            supervisor.time, supervisor.asyncio, supervisor.bootmgr, supervisor.write_reason = saved
        return marked, promoted, rt.info["trial"]

    marked, promoted, trial = run(failing=False)
    assert marked and 6000 <= marked[0] < 7000 and promoted and not trial, (marked, promoted, trial)
    marked, promoted, trial = run(failing=True)
    assert marked == [] and promoted == [] and trial, (marked, promoted, trial)


gc.collect()
print("%d passed, %d failed" % (passed[0], len(failures)))
if failures:
    print("failed:", " ".join(failures))
    sys.exit(1)
