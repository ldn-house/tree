"""The installed recovery layer, as the app sees it.

The recovery layer (bootmgr.py and friends) is only ever installed over USB,
so an OTA-installed app can find itself on an older one. The app's bundle
declares the oldest recovery API it runs on (RECOVERY_API in __init__.py;
bootmgr refuses a bundle that asks for more), and anything newer than that
is used when present and otherwise done here:

- API 3 added watchdog scratch 0 (scratch_set/scratch_get), deliberate()
  and the REASON-based reset_cause() that tells a hang ("watchdog") from a
  deliberate reset ("reboot"). On API 2 the app reads the same registers
  itself, best effort: an API 2 main.py doesn't clear the markers when
  someone takes over at the REPL, and reports every rp2 reset as "wdt".
"""

import os
import sys

import bootmgr

try:
    import machine
except ImportError:  # pragma: no cover
    machine = None

API = getattr(bootmgr, "API", 1)
MISSING = [n for n in ("scratch_set", "scratch_get", "deliberate", "watchdog_reason") if not hasattr(bootmgr, n)]
OUTDATED = bool(MISSING)
MESSAGE = "recovery layer outdated (USB install recommended)" if OUTDATED else None

# Same registers and encoding as bootmgr (API 3).
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
    if hasattr(bootmgr, "watchdog_reason"):
        return bootmgr.watchdog_reason()
    base = _watchdog_base()
    if base is None:
        return None
    return machine.mem32[base + _WD_REASON] & 3


def scratch_set(value):
    if hasattr(bootmgr, "scratch_set"):
        return bootmgr.scratch_set(value)
    base = _watchdog_base()
    if base is not None:
        machine.mem32[base + _WD_SCRATCH0] = 0 if value is None else _SCRATCH_MAGIC | (value & 0xFFFF)


def scratch_get():
    if hasattr(bootmgr, "scratch_get"):
        return bootmgr.scratch_get()
    base = _watchdog_base()
    if base is None:
        return None
    v = machine.mem32[base + _WD_SCRATCH0] & 0xFFFFFFFF
    if v & 0xFFF00000 != _SCRATCH_MAGIC:
        return None
    return v & 0xFFFF


def deliberate():
    """Forget the hang markers before an intentional reset (bootmgr.deliberate)."""
    if hasattr(bootmgr, "deliberate"):
        return bootmgr.deliberate()
    scratch_set(None)
    try:
        os.remove("fx_probe.txt")
    except OSError:
        pass


def reset():
    deliberate()
    bootmgr.reset()


def reset_cause(cause):
    """Refine the recovery layer's reset cause. API 3 already reports
    watchdog/reboot; API 2 says "wdt" for both, so read REASON here, and
    keep "wdt" if the platform can't say."""
    if cause == "wdt" and not hasattr(bootmgr, "watchdog_reason"):
        try:
            reason = watchdog_reason()
        except Exception:
            reason = None
        if reason is not None:
            if reason & 1:
                return "watchdog"
            if reason & 2:
                return "reboot"
    return cause


def status():
    return {"api": API, "outdated": OUTDATED, "missing": MISSING, "message": MESSAGE}
