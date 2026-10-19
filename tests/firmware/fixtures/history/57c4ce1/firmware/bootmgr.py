"""Boot manager and firmware installer: the recovery layer.

Installed over USB only (flash.py --usb). Nothing reachable over the network
can write this file, boot.py, main.py, safemode.py or config.py, so a bad
update can always be undone without USB.

Flash layout:

    /boot.py /main.py /bootmgr.py /safemode.py   recovery layer (this)
    /config.py                                   user settings (USB only)
    /boot.json                                   slot + boot counter
    /fw/a/  /fw/b/                               firmware slots
        manifest.json                            written last: slot is complete
        app/... effects/... data/...
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

Updates arrive as a single bundle (see Installer) and are written into the
inactive slot. Switching is one atomic rename of boot.json.
"""

import gc
import json
import os
import sys

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

API = 1
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


def write_atomic(path, data):
    """Write data to path so readers see the old or the new file, never a
    mix: LittleFS (and POSIX) rename replaces the target atomically."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    try:
        os.rename(tmp, path)
    except OSError:
        # Filesystems that refuse to rename over a file (FAT).
        try:
            os.remove(path)
        except OSError:
            pass
        os.rename(tmp, path)


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


def config_value(name, default=None):
    try:
        import config

        return getattr(config, name, default)
    except Exception:
        return default


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
    }


def read_state():
    st = _default_state()
    try:
        with open(STATE) as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in st:
                if k in data:
                    st[k] = data[k]
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
    return st


def write_state(st):
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
    if st["boots"] == 0 and not confirmed:
        return False
    st["boots"] = 0
    st["trial"] = False
    if confirmed:
        _event(st, "confirmed %s" % _ver(st["active"]))
    write_state(st)
    return confirmed


def commit(slot):
    """Make a fully installed slot active on the next boot, as a trial."""
    if not slot_ok(slot):
        raise InstallError("slot %s is not complete" % slot)
    st = read_state()
    if st["active"] != slot:
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


def reset():
    if machine is not None:
        machine.reset()
    raise SystemExit(0)


def reset_cause():
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


def record_crash(e):
    log("crash: %r" % (e,))
    try:
        with open(CRASH, "w") as f:
            sys.print_exception(e, f)
    except Exception:
        pass


def launch():
    """main.py: count the boot, choose a slot and run it."""
    st = read_state()
    kind, what = decide(st)
    st["boots"] += 1
    try:
        write_state(st)
    except Exception as e:
        log("cannot write boot state: %r" % (e,))
    info = {
        "slot": what if kind == "app" else None,
        "trial": st["trial"],
        "boots": st["boots"],
        "event": st["event"],
        "events": st["events"],
        "previous": st["previous"],
        "reset_cause": reset_cause(),
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
        # Someone is at the USB REPL: keep the watchdog fed for them.
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
    for _ in inst.finish(): ...; commit(inst.slot). Any problem raises
    InstallError and the caller should call abort(). Nothing outside the
    inactive slot is touched until commit().
    """

    def __init__(self, length, key=None, require_signature=None, allow_during_trial=False):
        st = read_state()
        if st["safe"] is None and st["trial"] and not allow_during_trial:
            raise InstallError("the running update is not confirmed yet")
        if not isinstance(length, int) or length <= len(MAGIC) + 8:
            raise InstallError("missing or too small Content-Length")
        self.length = length
        self.key = key
        # Default: with a key configured, every bundle must be signed.
        self.require_signature = bool(key) if require_signature is None else require_signature
        self.slot = other(st["active"]) if st["active"] else "a"
        self.dir = slot_dir(self.slot)
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
        if sig:
            if not self.key:
                # No key to check against; integrity still comes from hashes.
                return
            if not eq(sig, hmac_sha256(self.key, self.mbytes)):
                raise InstallError("bad signature")
            self.signed = True
        expect = len(MAGIC) + 4 + len(self.mbytes) + 4 + len(sig) + self.total
        if expect != self.length:
            raise InstallError("Content-Length does not match the manifest")

    def _start_files(self):
        rmtree(self.dir)
        gc.collect()
        try:
            st = os.statvfs("/" if exists("/") else ".")
            free = st[0] * st[3]
            if free < self.total + FREE_MARGIN:
                raise InstallError("not enough flash: need %d, free %d" % (self.total + FREE_MARGIN, free))
        except (AttributeError, OSError):
            pass
        makedirs(self.dir)
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
        try:
            rmtree(self.dir)
        except Exception:
            pass


def install_file(path, commit_after=True):
    """Install a bundle file already on the device (USB installs). Someone
    with USB access owns the board anyway, so an unsigned bundle is accepted
    here; a signature that is present must still be valid."""
    size = os.stat(path)[6]
    inst = Installer(size, config_value("UPDATE_KEY"), require_signature=False, allow_during_trial=True)
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
        commit(inst.slot)
    return inst
