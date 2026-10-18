"""Boot manager and firmware installer: the recovery layer.

Installed over USB (flash.py). No update, upload or API path writes this
file, boot.py, main.py or safemode.py, and config.py only changes through
the validated, signed config endpoint. That protects recovery against bad
or truncated updates. It does NOT protect it against code you choose to run:
firmware bundles and hot-pushed effects execute with full filesystem access,
so only install code signed with your UPDATE_KEY (see docs in CLAUDE.md,
"Trust model").

Flash layout:

    /boot.py /main.py /bootmgr.py /safemode.py   recovery layer (this)
    /config.py                                   user settings
    /config.good.json                            last config.py that loaded
    /boot.json                                   slot, boot counter, floor
    /fw/a/  /fw/b/                               firmware slots
        manifest.json                            written last: slot is complete
        app/... effects/... data/...
    /fw/new/                                     staging for an upload
    /fw/reserve.bin                              space kept back for recovery
    /fx/                                         hot-pushed effect overlays
    /state.json /coords.txt                      runtime data

Boot flow (launch(), called from main.py on every real boot):

1. Read boot.json and count the boot.
2. A slot that was just installed ("trial") gets one boot to report healthy.
   If the device reboots before that, roll back to the previous slot.
3. A confirmed slot that fails MAX_FAILED_BOOTS boots in a row falls back to
   the previous slot once, then enters safe mode (safemode.py): Wi-Fi plus a
   minimal authenticated HTTP server that accepts a new bundle.
4. Otherwise run app.runtime from the active slot. The app calls
   mark_healthy() once it has been up and responsive for a while, which
   resets the counter and confirms a trial slot.

Updates arrive as a single bundle (see Installer) and are written into
/fw/new, never into a slot. Only a fully verified bundle replaces the
inactive slot (commit()), and switching is one atomic rename of boot.json.
Bundles carry a signed, increasing "generation"; once a generation has been
confirmed healthy, older bundles are refused unless a downgrade is asked for
explicitly (anti-replay).
"""

import gc
import json
import os
import sys

import cfgcheck

try:
    import machine
except ImportError:  # pragma: no cover
    machine = None

try:
    import hashlib
except ImportError:  # pragma: no cover
    import uhashlib as hashlib

try:
    import binascii
except ImportError:  # pragma: no cover
    import ubinascii as binascii

API = 3  # 2: cfgcheck.py is part of the recovery layer; 3: scratch, deliberate(), REASON
WDT_MS = 8000
MAX_FAILED_BOOTS = 3

STATE = "boot.json"
FW = "fw"
SLOTS = ("a", "b")
FX_DIR = "fx"
LOG = "boot.log"
LOG_MAX = 8192
CRASH = "crash.txt"

MAGIC = b"TREEFW1\n"
MAX_MANIFEST = 16384
MAX_SIG = 64
ALLOWED_TOP = ("app", "effects", "lib", "data")
ALLOWED_EXT = (".py", ".mpy", ".json", ".txt", ".tcal")
REQUIRED = (("app/runtime.py", "app/runtime.mpy"),)
FREE_MARGIN = 32768
STAGE = FW + "/new"
RESERVE = FW + "/reserve.bin"
RESERVE_BYTES = 256 * 1024
RESERVE_HEADROOM = 128 * 1024  # free space to leave when (re)building the reserve
FLOOR_FILE = FW + "/floor.json"  # redundant copy of the anti-replay floor
CONFIG_GOOD = "config.good.json"  # last config confirmed to reach the network
CONFIG_PREV = "config.prev.py"  # config.py before an unconfirmed /config change
CONFIG_TYPES = (str, int, float, bool, type(None))

_wdt = None
_feeder = None


# --- small helpers -----------------------------------------------------------


def log(msg):
    print("[boot]", msg)
    try:
        try:
            if os.stat(LOG)[6] > LOG_MAX:
                os.rename(LOG, LOG + ".1")
        except OSError:
            pass
        with open(LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def exists(path):
    try:
        os.stat(path)
        return True
    except OSError:
        return False


def is_dir(path):
    try:
        return os.stat(path)[0] & 0x4000 != 0
    except OSError:
        return False


def makedirs(path):
    cur = ""
    for part in path.split("/"):
        if not part:
            continue
        cur = cur + "/" + part if cur else part
        if not exists(cur):
            os.mkdir(cur)


def rmtree(path):
    if not exists(path):
        return
    if is_dir(path):
        for entry in os.listdir(path):
            rmtree(path + "/" + entry)
        os.rmdir(path)
    else:
        os.remove(path)


def tree_size(path):
    """Bytes in the files under path (an estimate of what deleting it frees)."""
    if not exists(path):
        return 0
    if not is_dir(path):
        return os.stat(path)[6]
    return sum(tree_size(path + "/" + e) for e in os.listdir(path))


def write_atomic(path, data):
    """Write data to path so readers see the old or the new file, never a
    mix: LittleFS rename replaces the target atomically. Any failure leaves
    the old file untouched and raises; nothing here ever deletes it."""
    tmp = path + ".tmp"
    try:
        with open(tmp, "wb") as f:
            f.write(data)
        os.rename(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def free_bytes():
    try:
        st = os.statvfs("/" if exists("/") else ".")
        return st[0] * st[3]
    except (AttributeError, OSError):
        return None


def reclaim():
    """Free disposable space: staging, the recovery reserve, hot-pushed
    effects, rotated logs and stray temporary files. Returns bytes freed
    (None if unknown)."""
    before = free_bytes()
    for p in (STAGE, RESERVE, FX_DIR, LOG + ".1", "fx_probe.txt"):
        try:
            rmtree(p)
        except OSError:
            pass
    for d in ("", FW):
        try:
            for f in os.listdir(d or "."):
                if f.endswith(".tmp"):
                    rmtree((d + "/" if d else "") + f)
        except OSError:
            pass
    after = free_bytes()
    log("reclaimed space (free %s -> %s)" % (before, after))
    return None if before is None or after is None else after - before


def reserve_size():
    try:
        return os.stat(RESERVE)[6]
    except OSError:
        return 0


def reserve_steps():
    """Generator: (re)build the recovery reserve to RESERVE_BYTES, one 4 KB
    block per step, resuming a partial file (power cut mid-creation) and
    truncating an oversized one. Stops quietly if there isn't room for it
    plus RESERVE_HEADROOM. Async callers yield between steps."""
    have = reserve_size()
    if have > RESERVE_BYTES:
        try:
            os.remove(RESERVE)
        except OSError:
            return
        have = 0
    missing = RESERVE_BYTES - have
    if missing <= 0:
        return
    free = free_bytes()
    if free is None or free < missing + RESERVE_HEADROOM:
        return
    block = bytes(4096)
    try:
        makedirs(FW)
        with open(RESERVE, "ab") as f:
            while missing > 0:
                n = min(4096, missing)
                f.write(block[:n])
                missing -= n
                yield
    except OSError as e:
        log("cannot build reserve: %r" % (e,))


def ensure_reserve():
    for _ in reserve_steps():
        feed()


def eq(a, b):
    """Constant-time comparison of two byte strings."""
    if len(a) != len(b):
        return False
    r = 0
    for i in range(len(a)):
        r |= a[i] ^ b[i]
    return r == 0


def hmac_sha256(key, msg):
    if isinstance(key, str):
        key = key.encode()
    if len(key) > 64:
        key = hashlib.sha256(key).digest()
    key = key + bytes(64 - len(key))
    inner = hashlib.sha256(bytes(k ^ 0x36 for k in key))
    inner.update(msg)
    outer = hashlib.sha256(bytes(k ^ 0x5C for k in key))
    outer.update(inner.digest())
    return outer.digest()


def hexlify(b):
    return binascii.hexlify(b).decode()


def slot_dir(slot):
    return FW + "/" + slot


def other(slot):
    return "b" if slot == "a" else "a"


def slot_ok(slot):
    return slot in SLOTS and exists(slot_dir(slot) + "/manifest.json")


def slot_manifest(slot):
    try:
        with open(slot_dir(slot) + "/manifest.json") as f:
            return json.load(f)
    except Exception:
        return None


_config = None


def _read_good():
    try:
        with open(CONFIG_GOOD) as f:
            good = json.load(f)
    except Exception:
        return None
    clean, errors = cfgcheck.validate(good) if isinstance(good, dict) else ({}, ["bad"])
    return None if errors else clean


def load_config():
    """Settings from config.py, checked against cfgcheck.SCHEMA. If it is
    missing, fails to import or has any invalid setting, the last config
    confirmed to reach the network (config.good.json) is used instead, so
    the board stays reachable and can be fixed over Wi-Fi.
    Returns (values, source, error)."""
    global _config
    if _config is not None:
        return _config
    values = {}
    try:
        import config

        for k in dir(config):
            if k.isupper():
                values[k] = getattr(config, k)
        clean, errors = cfgcheck.validate(values)
        error = "; ".join(errors) if errors else None
    except BaseException as e:
        clean = {}
        error = "config.py failed to load: %r" % (e,)
    if error is None:
        _config = (clean, "config.py", None)
        return _config
    log("config: " + error)
    good = _read_good()
    if good is not None:
        log("using the last good config (config.good.json)")
        _config = (good, CONFIG_GOOD, error)
        return _config
    _config = (clean, "defaults", error)
    return _config


def config_value(name, default=None):
    return load_config()[0].get(name, default)


def config_source():
    return load_config()[1], load_config()[2]


def good_config():
    """The last config confirmed to reach the network, or None."""
    return _read_good()


def previous_config():
    """The config.py a /config change replaced (config.prev.py), checked
    against the schema, or None. Kept until the change is confirmed."""
    try:
        with open(CONFIG_PREV) as f:
            src = f.read()
        ns = {}
        exec(src, ns)
    except Exception:
        return None
    values = {k: v for k, v in ns.items() if k.isupper()}
    clean, errors = cfgcheck.validate(values)
    return None if errors else clean


def fallback_configs():
    """Configs whose Wi-Fi credentials are worth trying when the live ones
    don't work: the last good one, then the one a change replaced."""
    return [c for c in (_read_good(), previous_config()) if c]


def promote_config():
    """Called by the app once the running config is proven: Wi-Fi joined
    with its own credentials, HTTP serving and MQTT connected (when
    configured). Only then does it become the recovery fallback, and only
    then is a /config change confirmed. Returns True if anything changed."""
    values, source, _err = load_config()
    if source != "config.py":
        return False
    changed = False
    if _read_good() != values:
        write_atomic(CONFIG_GOOD, json.dumps(values).encode())
        changed = True
    st = read_state()
    if st["cfg_trial"]:
        st["cfg_trial"] = False
        st["cfg_boots"] = 0
        st["cfg_new"] = None
        _event(st, "config change confirmed")
        write_state(st)
        changed = True
    try:
        os.remove(CONFIG_PREV)
    except OSError:
        pass
    return changed


def install_config(values, trial=True):
    """Replace config.py from plain values, as a trial: the old config.py is
    kept (config.prev.py) and the new one is only promoted to the recovery
    fallback once the app reaches the network with it; a boot that doesn't
    confirm reverts it. Every setting must pass cfgcheck (strict: unknown
    names are refused) and UPDATE_KEY can't be changed or removed this way
    (USB only), so a signed config can't switch signing off. No code is
    accepted: the file is generated from validated literals. Raises
    ValueError."""
    if not isinstance(values, dict):
        raise ValueError("expected an object")
    clean, errors = cfgcheck.validate(values, strict=True)
    if errors:
        raise ValueError("; ".join(errors))
    if clean.get("UPDATE_KEY") != config_value("UPDATE_KEY"):
        raise ValueError("UPDATE_KEY can only be changed over USB")
    st = read_state()
    if st["trial"] and st["safe"] is None:
        raise ValueError("a firmware update is still on trial; try again once it is confirmed")
    if st["cfg_trial"] and trial:
        raise ValueError("the previous config change is not confirmed yet")
    lines = ["# Written by the device from a validated /config upload.\n"]
    for k in sorted(clean):
        lines.append("%s = %r\n" % (k, clean[k]))
    src = "".join(lines)
    compile(src, "config.py", "exec")
    new = src.encode()
    # Order matters for power cuts (review R3-H1). 1: keep the old config
    # (or make sure no stale copy from an earlier change is left to be
    # "restored"). 2: record the change, and which file it installs, in
    # boot.json. 3: only then replace config.py. A cut before 3 leaves the
    # old config live, and launch() sees config.py isn't the recorded one
    # and drops the change; a cut after 3 is an ordinary trial, reverted
    # unless it reaches the network. From safe mode (trial=False) there is
    # no app boot to confirm it; the old config stays as config.prev.py, and
    # its credentials are a Wi-Fi fallback (fallback_configs()).
    if exists("config.py"):
        with open("config.py", "rb") as f:
            old = f.read()
        write_atomic(CONFIG_PREV, old)
    elif exists(CONFIG_PREV):
        os.remove(CONFIG_PREV)
    st["cfg_trial"] = trial
    st["cfg_boots"] = 0
    st["cfg_new"] = _digest(new) if trial else None
    _event(st, "config change installed" + (" (trial)" if trial else " from safe mode"))
    write_state(st)
    write_atomic("config.py", new)
    return sorted(clean)


def _digest(data):
    return hexlify(hashlib.sha256(data).digest())[:32]


def _file_digest(path):
    try:
        with open(path, "rb") as f:
            return _digest(f.read())
    except OSError:
        return None


def _revert_config(st, why):
    """Undo an unconfirmed /config change: restore config.py from before it
    (or, without one, drop config.py so the last good config is used)."""
    try:
        if exists(CONFIG_PREV):
            os.rename(CONFIG_PREV, "config.py")
        else:
            os.remove("config.py")
    except OSError as e:
        log("cannot revert config: %r" % (e,))
    st["cfg_trial"] = False
    st["cfg_boots"] = 0
    st["cfg_new"] = None
    _event(st, "config change reverted: " + why)


def signature_ok(kind, body, sig_hex, key=None):
    """Check an HMAC-SHA256 over b"tree-<kind>\n" + body (domain-separated
    from bundle manifests) made with UPDATE_KEY. True if no key is set."""
    key = key if key is not None else config_value("UPDATE_KEY")
    if not key:
        return True
    if not sig_hex:
        return False
    try:
        sig = binascii.unhexlify(sig_hex.strip())
    except ValueError:
        return False
    return eq(sig, hmac_sha256(key, b"tree-" + kind.encode() + b"\n" + body))


# --- watchdog ----------------------------------------------------------------


def watchdog():
    """The hardware watchdog (started on first use; it cannot be stopped)."""
    global _wdt
    if _wdt is None and machine is not None:
        _wdt = machine.WDT(timeout=WDT_MS)
    return _wdt


def feed():
    if _wdt is not None:
        _wdt.feed()


def _feed_cb(_t):
    feed()


def start_feeder():
    """Feed the watchdog from a soft timer, for when no firmware is running
    (USB sessions). The app stops this and feeds from its supervisor."""
    global _feeder
    if machine is None or _feeder is not None:
        return
    watchdog()
    try:
        _feeder = machine.Timer(period=1000, mode=machine.Timer.PERIODIC, callback=_feed_cb)
    except Exception as e:
        log("feeder timer failed: %r" % (e,))


def stop_feeder():
    global _feeder
    if _feeder is not None:
        _feeder.deinit()
        _feeder = None


def early():
    """boot.py: keep the watchdog fed until main.py (or a USB session) takes
    over. Starting it here means every later stage runs under it."""
    start_feeder()


# --- boot state ----------------------------------------------------------------


def _default_state():
    return {
        "active": None,
        "previous": None,
        "trial": False,
        "boots": 0,
        "bad": None,
        "safe": None,
        "event": None,
        "events": [],
        "floor": 0,  # lowest bundle generation accepted (anti-replay)
        "cfg_trial": False,  # a /config change awaiting confirmation
        "cfg_boots": 0,
        "cfg_new": None,  # digest of the config.py that change installs
    }


def _valid_floor(v):
    return isinstance(v, int) and not isinstance(v, bool) and v >= 0


def _floor_file():
    try:
        with open(FLOOR_FILE) as f:
            v = json.load(f)
        return v if _valid_floor(v) else None
    except Exception:
        return None


def read_state():
    st = _default_state()
    stored_floor = None  # only a floor actually present in boot.json counts
    try:
        with open(STATE) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in st:
                if k in data:
                    st[k] = data[k]
            if _valid_floor(data.get("floor")):
                stored_floor = data["floor"]
    except Exception:
        pass
    if not isinstance(st["boots"], int) or st["boots"] < 0:
        st["boots"] = 0
    if st["active"] not in SLOTS:
        st["active"] = None
    if st["previous"] not in SLOTS:
        st["previous"] = None
    if not isinstance(st["events"], list):
        st["events"] = []
    if not isinstance(st["cfg_boots"], int) or st["cfg_boots"] < 0:
        st["cfg_boots"] = 0
    st["cfg_trial"] = st["cfg_trial"] is True
    if not isinstance(st["cfg_new"], str):
        st["cfg_new"] = None
    # Anti-replay floor: the highest of boot.json and its redundant copy. If
    # neither is readable, don't assume "nothing was ever confirmed" (that
    # would accept any old signed bundle): fall back to the newest installed
    # slot's generation. An explicit downgrade still gets past it.
    floor = stored_floor
    copy = _floor_file()
    if floor is None and copy is None:
        floor = max([generation(slot_manifest(s)) for s in SLOTS] + [0])
        st["floor_source"] = "slots"
    else:
        floor = max(floor or 0, copy or 0)
    st["floor"] = floor
    return st


def write_state(st):
    st = dict(st)
    st.pop("floor_source", None)
    write_atomic(STATE, json.dumps(st).encode())


def _event(st, msg):
    log(msg)
    st["event"] = msg
    ev = st["events"][-4:]
    ev.append(msg)
    st["events"] = ev


def decide(st):
    """Choose what to boot. Mutates st; returns ("app", slot) or ("safe", why)."""
    if st["safe"]:
        return ("safe", st["safe"])
    slot = st["active"]
    if not slot_ok(slot):
        alt = other(slot) if slot else "a"
        if not slot_ok(alt):
            alt = other(alt)
        if slot_ok(alt):
            if slot is not None:
                _event(st, "slot %s is incomplete, booting %s" % (slot, alt))
            st["active"] = alt
            st["previous"] = None
            st["trial"] = False
            st["boots"] = 0
            slot = alt
        else:
            st["safe"] = "no firmware installed"
            _event(st, "safe mode: no firmware installed")
            return ("safe", st["safe"])

    prev = st["previous"]
    prev_ok = prev is not None and prev != slot and slot_ok(prev)
    if st["trial"] and st["boots"] >= 1:
        st["trial"] = False
        if prev_ok:
            _event(st, "rolled back %s -> %s: update did not confirm health" % (_ver(slot), _ver(prev)))
            st["active"] = prev
            st["previous"] = slot
            st["bad"] = slot
            st["boots"] = 0
            slot = prev
        else:
            _event(st, "update %s unconfirmed and nothing to roll back to" % _ver(slot))
    elif st["boots"] >= MAX_FAILED_BOOTS:
        if prev_ok and st["bad"] != prev:
            _event(st, "fell back %s -> %s after %d failed boots" % (_ver(slot), _ver(prev), st["boots"]))
            st["active"] = prev
            st["previous"] = slot
            st["bad"] = slot
            st["boots"] = 0
            slot = prev
        else:
            st["safe"] = "%d boots without reaching health" % st["boots"]
            _event(st, "safe mode: " + st["safe"])
            return ("safe", st["safe"])
    return ("app", slot)


def _ver(slot):
    m = slot_manifest(slot)
    v = m.get("version") if isinstance(m, dict) else None
    return "%s(%s)" % (slot, v) if v else slot


def mark_healthy(slot):
    """Called by the app once it is healthy: resets the boot counter and
    confirms a trial slot. Only the slot that is actually running may do
    this (an update committed but not yet booted must not be confirmed by
    the old firmware). Returns True if this confirmed an update."""
    st = read_state()
    if st["active"] != slot:
        return False
    confirmed = st["trial"]
    gen = generation(slot_manifest(slot))
    if st["boots"] or confirmed or gen > st["floor"]:
        st["boots"] = 0
        st["trial"] = False
        if confirmed:
            _event(st, "confirmed %s" % _ver(st["active"]))
        if gen > st["floor"]:
            st["floor"] = gen
        write_state(st)
    if _floor_file() != st["floor"]:
        try:
            write_atomic(FLOOR_FILE, json.dumps(st["floor"]).encode())
        except OSError as e:
            log("cannot save floor copy: %r" % (e,))
    return confirmed


def generation(manifest):
    g = manifest.get("generation") if isinstance(manifest, dict) else None
    return g if isinstance(g, int) and not isinstance(g, bool) and g > 0 else 0


def install_refused():
    """Why a (non-dry-run) install can't proceed now, or None."""
    st = read_state()
    if st["trial"] and st["safe"] is None:
        return "the running update is not confirmed yet"
    return None


def commit(allow_during_trial=False):
    """Move the verified staging tree into the inactive slot and boot it
    next, as a trial. This is the only place a slot is ever replaced, and it
    is refused while the running firmware is itself an unconfirmed trial
    (its rollback target is the slot that would be overwritten). Returns the
    slot."""
    if not exists(STAGE + "/manifest.json"):
        raise InstallError("nothing staged")
    st = read_state()
    if st["trial"] and st["safe"] is None and not allow_during_trial:
        raise InstallError("the running update is not confirmed yet")
    slot = other(st["active"]) if st["active"] else "a"
    if st["previous"] == slot:
        # Detach the slot about to be replaced first, so a power cut part way
        # through never leaves boot.json pointing at a half-swapped slot.
        st["previous"] = None
        write_state(st)
    rmtree(slot_dir(slot))
    os.rename(STAGE, slot_dir(slot))
    if st["active"] != slot and st["active"] is not None:
        st["previous"] = st["active"]
    st["active"] = slot
    st["trial"] = True
    st["boots"] = 0
    st["safe"] = None
    if st["bad"] == slot:
        st["bad"] = None
    _event(st, "installed %s (trial)" % _ver(slot))
    write_state(st)
    # Hot-pushed effects were written against the old firmware.
    try:
        rmtree(FX_DIR)
    except OSError:
        pass
    return slot


def rollback():
    """Switch to the previous slot on the next boot. Returns the slot or None."""
    st = read_state()
    prev = st["previous"]
    if prev is None or prev == st["active"] or not slot_ok(prev):
        return None
    _event(st, "manual rollback %s -> %s" % (_ver(st["active"]), _ver(prev)))
    st["previous"] = st["active"]
    st["active"] = prev
    st["trial"] = False
    st["boots"] = 0
    st["safe"] = None
    write_state(st)
    return prev


def leave_safe_mode():
    st = read_state()
    st["safe"] = None
    st["boots"] = 0
    st["bad"] = None
    _event(st, "leaving safe mode")
    write_state(st)


# --- resets: deliberate or watchdog? ------------------------------------------
#
# On the rp2 port machine.reset() is watchdog_reboot(): it forces a watchdog
# reset, so machine.reset_cause() reports WDT_RESET after *every* deliberate
# reset (mpremote's reset, flash.py, /reboot, post-update reboots) as well
# as after a real hang. The watchdog's REASON register tells them apart:
# TIMER (bit 0) when the watchdog expired, FORCE (bit 1) when it was
# triggered. Watchdog scratch 0 holds the running effect (see app.engine).

_WD_REASON = 0x08
_WD_SCRATCH0 = 0x0C
_SCRATCH_MAGIC = 0x7EE00000


def _watchdog_base():
    if machine is None or not hasattr(machine, "mem32"):
        return None
    base = getattr(machine, "DEVSIM_WATCHDOG_BASE", None)  # the unix-port simulator
    if base:
        return base
    chip = getattr(sys.implementation, "_machine", "")
    if "RP2350" in chip:
        return 0x400D8000
    if "RP2040" in chip:
        return 0x40058000
    return None


def watchdog_reason():
    """REASON bits for the last reset: 1 timer (expired), 2 force, 0 neither
    (power-on, RUN pin), None if unknown on this platform."""
    base = _watchdog_base()
    if base is None:
        return None
    return machine.mem32[base + _WD_REASON] & 3


def scratch_set(value):
    """Record a 16-bit value (or None to clear) in watchdog scratch 0; it
    survives watchdog and forced resets but not a power cycle."""
    base = _watchdog_base()
    if base is not None:
        machine.mem32[base + _WD_SCRATCH0] = 0 if value is None else _SCRATCH_MAGIC | (value & 0xFFFF)


def scratch_get():
    base = _watchdog_base()
    if base is None:
        return None
    v = machine.mem32[base + _WD_SCRATCH0] & 0xFFFFFFFF
    if v & 0xFFF00000 != _SCRATCH_MAGIC:
        return None
    return v & 0xFFFF


def deliberate():
    """Before any intentional reset (or when someone takes over at the USB
    REPL): forget which effect was running, so the next boot can't mistake
    this reset for that effect hanging the board."""
    scratch_set(None)
    try:
        os.remove("fx_probe.txt")
    except OSError:
        pass


def reset():
    deliberate()
    if machine is not None:
        machine.reset()
    raise SystemExit(0)


def machine_reset_cause():
    """machine.reset_cause() by name: pwron, hard, wdt, deepsleep, soft. This
    is what launch() passes apps as info["reset_cause"]: apps written for
    API 2 quarantine a hanging effect on "wdt" (review R3-M2)."""
    if machine is None:
        return "unknown"
    try:
        c = machine.reset_cause()
    except Exception:
        return "unknown"
    for name in ("PWRON_RESET", "HARD_RESET", "WDT_RESET", "DEEPSLEEP_RESET", "SOFT_RESET"):
        if getattr(machine, name, None) == c:
            return name[:-6].lower()
    return str(c)


def reset_cause():
    """pwron, watchdog (it expired: something hung), reboot (a deliberate
    machine.reset()), or wdt if the platform can't say which. Apps get it
    as info["reset_kind"]."""
    cause = machine_reset_cause()
    if cause == "wdt":
        reason = watchdog_reason()
        if reason is not None:
            if reason & 1:
                return "watchdog"
            if reason & 2:
                return "reboot"
    return cause


def record_crash(e):
    log("crash: %r" % (e,))
    try:
        with open(CRASH, "w") as f:
            sys.print_exception(e, f)
    except Exception:
        pass


def config_boot(st, kind):
    """Config-trial bookkeeping for a boot (mutates st; the caller writes it).
    kind is what decide() chose: "app" or "safe"."""
    if st["cfg_trial"] and st["cfg_new"] is not None and _file_digest("config.py") != st["cfg_new"]:
        # The change was recorded but config.py isn't the file it installs:
        # power was lost before the new file landed, or after a revert put
        # the old one back. Either way the live config is the old one.
        st["cfg_trial"] = False
        st["cfg_boots"] = 0
        st["cfg_new"] = None
        try:
            if exists("config.py") and exists(CONFIG_PREV):
                os.remove(CONFIG_PREV)
        except OSError:
            pass
        _event(st, "config change not applied (interrupted); keeping the current config")
    if st["cfg_trial"]:
        if st["cfg_boots"] >= 1 or kind == "safe":
            # The config change didn't confirm on its boot (or we're going
            # to safe mode, which can't confirm one): put the old one back.
            _revert_config(st, "it did not reach the network" if kind == "app" else "entering safe mode")
        else:
            st["cfg_boots"] += 1


def launch():
    """main.py: count the boot, choose a slot and run it."""
    st = read_state()
    kind, what = decide(st)
    config_boot(st, kind)
    st["boots"] += 1
    try:
        write_state(st)
    except Exception as e:
        # Without a persisted counter, rollback and the three-strike rule
        # can't work, and a failing app would retry forever. Go straight to
        # recovery, decided in RAM; safe mode reclaims space to install.
        log("cannot write boot state: %r" % (e,))
        kind, what = "safe", "cannot record the boot (%r)" % (e,)
    info = {
        "slot": what if kind == "app" else None,
        "trial": st["trial"],
        "boots": st["boots"],
        "event": st["event"],
        "events": st["events"],
        "previous": st["previous"],
        "reset_cause": machine_reset_cause(),  # the API 2 contract
        "reset_kind": reset_cause(),  # API 3: watchdog / reboot told apart
        "cfg_trial": st["cfg_trial"],
    }
    watchdog()
    stop_feeder()
    feed()
    try:
        if kind == "safe":
            import safemode

            safemode.run(what, info)
        else:
            path = slot_dir(what)
            sys.path.insert(0, path)
            info["manifest"] = slot_manifest(what)
            gc.collect()
            from app import runtime

            runtime.run(info)
    except KeyboardInterrupt:
        # Someone is at the USB REPL: keep the watchdog fed for them, and
        # forget the running effect (mpremote's reset is coming, and it is
        # not a hang).
        deliberate()
        start_feeder()
        print("[boot] interrupted; watchdog fed by timer. machine.reset() to restart.")
        return
    except BaseException as e:
        record_crash(e)
    reset()


# --- installer -------------------------------------------------------------------


class InstallError(Exception):
    pass


def check_path(p):
    if not isinstance(p, str) or not p or len(p) > 64:
        return False
    if p[0] == "/" or "//" in p or ".." in p:
        return False
    for ch in p:
        if not (ch.isalpha() and ch.lower() == ch or ch.isdigit() or ch in "_./-"):
            return False
    parts = p.split("/")
    if len(parts) > 4 or parts[0] not in ALLOWED_TOP:
        return False
    for ext in ALLOWED_EXT:
        if p.endswith(ext):
            return True
    return False


def check_mpy(header):
    """Raise unless an .mpy header can be loaded by this interpreter."""
    want = getattr(sys.implementation, "_mpy", None)
    if len(header) < 4 or header[0] != 0x4D:  # "M"
        raise InstallError("not an .mpy file")
    if want is None:
        return
    if header[1] != want & 0xFF:
        raise InstallError("mpy version %d, device needs %d" % (header[1], want & 0xFF))
    # Byte 2 is only set for files with native code: then the sub-version
    # and architecture must match too, as the loader requires.
    if header[2]:
        if (header[2] & 3) != (want >> 8) & 3:
            raise InstallError("mpy sub-version %d, device needs %d" % (header[2] & 3, (want >> 8) & 3))
        if header[2] >> 2 != (want >> 10) & 0x3F:
            raise InstallError("mpy built for another architecture")


def check_source(path, src):
    """Compile-check Python source (catches truncation and syntax errors)."""
    try:
        compile(src, path, "exec")
    except NameError:  # compile() not enabled in this build
        pass
    except SyntaxError as e:
        raise InstallError("%s: syntax error: %s" % (path, e))
    except MemoryError:
        raise InstallError("%s: out of memory compiling" % path)


_P_HEAD, _P_MANIFEST, _P_SIGLEN, _P_SIG, _P_FILES, _P_DONE = range(6)


class Installer:
    """Streams a firmware bundle into the inactive slot and verifies it.

    Bundle layout:
        b"TREEFW1\\n"
        u32 big-endian manifest length, manifest JSON
        u32 big-endian signature length (0 or 32), HMAC-SHA256(key, manifest)
        file contents, concatenated in manifest order

    manifest = {"format": 1, "version": ..., "bootmgr": 1,
                "files": [{"path": "app/runtime.mpy", "size": n, "sha256": hex}, ...]}

    Use: inst = Installer(content_length, key); inst.feed(chunk)...;
    for _ in inst.finish(): ...; commit(). Any problem raises InstallError and
    the caller should call abort(). Files only ever go to the staging tree
    (/fw/new), which abort() removes only if this installer created it: a
    rejected or dry-run upload never touches either slot.

    manifest "generation" (signed with the rest of the manifest) must be at
    least the persisted floor, so a captured older bundle can't be replayed,
    unless allow_downgrade (an explicit, authenticated request).
    """

    def __init__(self, length, key=None, require_signature=None, allow_downgrade=False, reclaim_slot=False):
        st = read_state()
        if not isinstance(length, int) or length <= len(MAGIC) + 8:
            raise InstallError("missing or too small Content-Length")
        self.length = length
        self.key = key
        # Default: with a key configured, every bundle must be signed.
        self.require_signature = bool(key) if require_signature is None else require_signature
        self.floor = st["floor"]
        self.allow_downgrade = allow_downgrade
        # Safe mode only: if the flash is too full, the inactive slot (which
        # failed to boot or is about to be replaced) may be sacrificed.
        self.reclaim_slot = reclaim_slot
        self.inactive = other(st["active"]) if st["active"] else "a"
        self.dir = STAGE
        self.created = False
        self.got = 0
        self.phase = _P_HEAD
        self.need = len(MAGIC) + 4
        self.buf = bytearray()
        self.manifest = None
        self.mbytes = None
        self.signed = False
        self.files = []
        self.idx = -1
        self.f = None
        self.h = None
        self.left = 0

    # -- streaming --

    def feed(self, data):
        n = len(data)
        self.got += n
        if self.got > self.length:
            raise InstallError("more data than Content-Length")
        mv = memoryview(data)
        pos = 0
        while pos < n:
            if self.phase == _P_FILES:
                take = min(self.left, n - pos)
                if take:
                    chunk = mv[pos : pos + take]
                    self.f.write(chunk)
                    self.h.update(chunk)
                    self.left -= take
                    pos += take
                if self.left == 0:
                    self._close_file()
                continue
            if self.phase == _P_DONE:
                raise InstallError("trailing data after last file")
            take = min(self.need - len(self.buf), n - pos)
            self.buf.extend(mv[pos : pos + take])
            pos += take
            if len(self.buf) == self.need:
                self._header_step()

    def _header_step(self):
        b = bytes(self.buf)
        self.buf = bytearray()
        if self.phase == _P_HEAD:
            if b[: len(MAGIC)] != MAGIC:
                raise InstallError("not a firmware bundle")
            mlen = int.from_bytes(b[len(MAGIC) :], "big")
            if not 2 <= mlen <= MAX_MANIFEST:
                raise InstallError("bad manifest length")
            self.phase = _P_MANIFEST
            self.need = mlen
        elif self.phase == _P_MANIFEST:
            self.mbytes = b
            self._check_manifest(b)
            self.phase = _P_SIGLEN
            self.need = 4
        elif self.phase == _P_SIGLEN:
            slen = int.from_bytes(b, "big")
            if slen > MAX_SIG:
                raise InstallError("bad signature length")
            if slen == 0:
                self._check_signature(b"")
                self._start_files()
            else:
                self.phase = _P_SIG
                self.need = slen
        elif self.phase == _P_SIG:
            self._check_signature(b)
            self._start_files()

    def _check_manifest(self, b):
        try:
            m = json.loads(b)
        except ValueError:
            raise InstallError("manifest is not valid JSON")
        if not isinstance(m, dict) or m.get("format") != 1:
            raise InstallError("unsupported manifest format")
        if not isinstance(m.get("bootmgr", 1), int) or m.get("bootmgr", 1) > API:
            raise InstallError("bundle needs a newer recovery layer (install over USB)")
        files = m.get("files")
        if not isinstance(files, list) or not files:
            raise InstallError("manifest lists no files")
        seen = {}
        total = 0
        for e in files:
            if not isinstance(e, dict):
                raise InstallError("bad file entry")
            p = e.get("path")
            size = e.get("size")
            digest = e.get("sha256")
            if not check_path(p) or p in seen:
                raise InstallError("bad or duplicate path: %r" % (p,))
            if not isinstance(size, int) or size < 0:
                raise InstallError("bad size for " + p)
            if not isinstance(digest, str) or len(digest) != 64:
                raise InstallError("bad sha256 for " + p)
            try:
                binascii.unhexlify(digest)
            except ValueError:
                raise InstallError("bad sha256 for " + p)
            seen[p] = 1
            total += size
        for alternatives in REQUIRED:
            if not any(a in seen for a in alternatives):
                raise InstallError("bundle has no " + alternatives[0])
        self.manifest = m
        self.files = files
        self.total = total
        # The whole upload length is known now: reject truncated or padded
        # uploads before writing anything.
        header = len(MAGIC) + 4 + len(b) + 4
        if self.length < header + total or self.length > header + MAX_SIG + total:
            raise InstallError("Content-Length does not match the manifest")

    def _check_signature(self, sig):
        if self.require_signature:
            if not self.key:
                raise InstallError("signed bundles required but no UPDATE_KEY configured")
            if not sig:
                raise InstallError("bundle is not signed")
        if sig and self.key:
            if not eq(sig, hmac_sha256(self.key, self.mbytes)):
                raise InstallError("bad signature")
            self.signed = True
        # Without a key a signature can't be checked; integrity still comes
        # from the hashes, and the length and generation checks below apply
        # to every bundle, signed or not.
        expect = len(MAGIC) + 4 + len(self.mbytes) + 4 + len(sig) + self.total
        if expect != self.length:
            raise InstallError("Content-Length does not match the manifest")
        # Only now is the manifest authenticated (when a key is set), so only
        # now is its generation worth acting on. Still before any write.
        gen = generation(self.manifest)
        if gen < self.floor and not self.allow_downgrade:
            raise InstallError(
                "bundle generation %d is older than the minimum %d already confirmed (a replay or downgrade);"
                " ask for an explicit downgrade to install it" % (gen, self.floor)
            )

    def _start_files(self):
        # Staging is disposable by definition: a leftover tree from an
        # interrupted upload is cleared. Slots are never touched here.
        rmtree(STAGE)
        gc.collect()
        need = self.total + FREE_MARGIN
        free = free_bytes()
        if free is not None and free < need:
            reclaim()
            free = free_bytes()
            if free is not None and free < need and self.reclaim_slot:
                # Explicitly requested (safe mode, ?reclaim_slot=1): sacrifice
                # the inactive slot, but only if that actually makes room.
                if free + tree_size(slot_dir(self.inactive)) < need:
                    raise InstallError("not enough flash even without slot %s: need %d" % (self.inactive, need))
                log("freeing inactive slot %s for the upload (requested)" % self.inactive)
                st = read_state()
                if st["previous"] == self.inactive:
                    st["previous"] = None
                    write_state(st)
                rmtree(slot_dir(self.inactive))
                free = free_bytes()
            if free is not None and free < need:
                raise InstallError("not enough flash: need %d, free %d" % (need, free))
        makedirs(self.dir)
        self.created = True
        self.phase = _P_FILES
        self._next_file()

    def _next_file(self):
        self.idx += 1
        if self.idx >= len(self.files):
            self.phase = _P_DONE
            return
        e = self.files[self.idx]
        path = self.dir + "/" + e["path"]
        d = path.rsplit("/", 1)[0]
        makedirs(d)
        self.f = open(path, "wb")
        self.h = hashlib.sha256()
        self.left = e["size"]
        if self.left == 0:
            self._close_file()

    def _close_file(self):
        e = self.files[self.idx]
        self.f.close()
        self.f = None
        if hexlify(self.h.digest()) != e["sha256"].lower():
            raise InstallError("sha256 mismatch for " + e["path"])
        self.h = None
        self._next_file()

    # -- verification and abort --

    def finish(self):
        """Generator: verifies the installed files, yielding between files so
        an async caller can keep the event loop (and watchdog) going."""
        if self.phase != _P_DONE or self.got != self.length:
            raise InstallError("upload truncated (%d of %d bytes)" % (self.got, self.length))
        for e in self.files:
            path = self.dir + "/" + e["path"]
            if e["path"].endswith(".py"):
                with open(path) as f:
                    src = f.read()
                check_source(e["path"], src)
                src = None
                gc.collect()
            elif e["path"].endswith(".mpy"):
                with open(path, "rb") as f:
                    check_mpy(f.read(4))
            yield
        write_atomic(self.dir + "/manifest.json", self.mbytes)
        yield

    def abort(self):
        try:
            if self.f is not None:
                self.f.close()
                self.f = None
        except Exception:
            pass
        if self.created:
            try:
                rmtree(self.dir)
            except Exception:
                pass
            self.created = False


def install_file(path, commit_after=True):
    """Install a bundle file already on the device (USB installs). Someone
    with USB access owns the board anyway, so an unsigned bundle is accepted
    here; a signature that is present must still be valid."""
    size = os.stat(path)[6]
    inst = Installer(size, config_value("UPDATE_KEY"), require_signature=False, allow_downgrade=True)
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(2048)
                if not chunk:
                    break
                inst.feed(chunk)
                feed()
        for _ in inst.finish():
            feed()
    except BaseException:
        inst.abort()
        raise
    if commit_after:
        inst.slot = commit(allow_during_trial=True)
    return inst
