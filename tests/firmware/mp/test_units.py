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
        "bootmgr": 1,
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


def install(data, key=None, chunk=7, require=None, length=None):
    inst = bootmgr.Installer(len(data) if length is None else length, key, require_signature=require,
                             allow_during_trial=True)
    try:
        for i in range(0, len(data), chunk):
            inst.feed(data[i : i + chunk])
        for _ in inst.finish():
            pass
    except Exception:
        inst.abort()
        raise
    return inst


@test
def installer_good_bundle_and_commit():
    reset_flash()
    inst = install(make_bundle(GOOD))
    assert inst.slot == "a"
    assert bootmgr.slot_ok("a")
    with open("fw/a/app/x.py") as f:
        assert f.read() == "X = 1\n"
    bootmgr.commit("a")
    st = bootmgr.read_state()
    assert st["active"] == "a" and st["trial"] and st["boots"] == 0


@test
def installer_truncated_upload_is_rejected():
    reset_flash()
    data = make_bundle(GOOD)
    inst = bootmgr.Installer(len(data), None, allow_during_trial=True)
    inst.feed(data[: len(data) - 5])
    raises(lambda: list(inst.finish()), "truncated")
    inst.abort()
    assert not bootmgr.slot_ok("a") and not bootmgr.exists("fw/a")


@test
def installer_length_mismatch_rejected_before_writing():
    reset_flash()
    data = make_bundle(GOOD)
    raises(lambda: install(data, length=len(data) + 100), "Content-Length")
    assert not bootmgr.exists("fw/a")


@test
def installer_bad_hash():
    reset_flash()
    data = bytearray(make_bundle(GOOD))
    data[-12] ^= 0x01  # inside app/x.py's body (data/a.txt is empty)
    raises(lambda: install(bytes(data)), "sha256 mismatch")
    assert not bootmgr.slot_ok("a")


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
    install(make_bundle(GOOD))
    bootmgr.commit("a")
    raises(lambda: bootmgr.Installer(100, None), "not confirmed")
    bootmgr.mark_healthy("a")
    inst = install(make_bundle(GOOD, version="t2"))
    assert inst.slot == "b"
    bootmgr.commit("b")
    st = bootmgr.read_state()
    assert st["active"] == "b" and st["previous"] == "a"


# -- boot decisions ------------------------------------------------------------


def two_slots():
    reset_flash()
    install(make_bundle(GOOD, version="v1"))
    bootmgr.commit("a")
    bootmgr.mark_healthy("a")
    install(make_bundle(GOOD, version="v2"))
    bootmgr.commit("b")


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
    install(make_bundle(GOOD))
    st = bootmgr.read_state()  # no boot.json yet
    assert bootmgr.decide(st) == ("app", "a")
    os.remove("fw/a/manifest.json")
    st = {"active": "a", "previous": None, "trial": False, "boots": 0, "bad": None, "safe": None,
          "event": None, "events": []}
    assert bootmgr.decide(st)[0] == "safe"


@test
def boot_json_corrupt_is_recovered():
    reset_flash()
    install(make_bundle(GOOD))
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


# -- output stage --------------------------------------------------------------


@test
def output_cap_applies_before_brightness():
    from app import outstage

    n = 500
    out = outstage.create(n, 13, None, "RGB")
    src = bytearray([255] * (3 * n))
    dst = bytearray(3 * n)
    out.process(src, dst, 255)
    assert out.capped and sum(dst) <= out.budget
    full = sum(dst)
    out.process(src, dst, 128)
    half = sum(dst)
    assert abs(half - full * 128 // 255) <= 3 * n, (full, half)  # brightness still dims when capped
    sparse = bytearray(3 * n)
    for i in range(0, 3 * n, 30):
        sparse[i] = 255
    out.process(sparse, dst, 255)
    assert not out.capped and dst == sparse


@test
def output_colour_order():
    from app import outstage

    out = outstage.create(2, 100, None, "GRB")
    dst = bytearray(6)
    out.process(bytearray([10, 20, 30, 1, 2, 3]), dst, 255)
    assert list(dst) == [20, 10, 30, 2, 1, 3]


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


gc.collect()
print("%d passed, %d failed" % (passed[0], len(failures)))
if failures:
    print("failed:", " ".join(failures))
    sys.exit(1)
