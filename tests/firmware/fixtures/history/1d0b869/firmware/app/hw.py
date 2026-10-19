"""Platform shims so the same code runs on the Pico and the unix port."""

import gc
import sys

try:
    import machine
except ImportError:  # pragma: no cover
    machine = None

try:
    import binascii
except ImportError:  # pragma: no cover
    import ubinascii as binascii

# Watchdog scratch register 0. It survives a watchdog reset (but not a power
# cycle), so the runtime records the running effect there and can tell which
# effect was on screen when the watchdog fired. Scratch 4-7 belong to the SDK
# and bootrom.
_SCRATCH0 = None
_mach = getattr(sys.implementation, "_machine", "")
if machine is not None and hasattr(machine, "mem32"):
    if "RP2350" in _mach:
        _SCRATCH0 = 0x400D8000 + 0x0C
    elif "RP2040" in _mach:
        _SCRATCH0 = 0x40058000 + 0x0C

_SCRATCH_MAGIC = 0x7EE00000


def scratch_set(value):
    """Store a 16-bit value (or None to clear) in watchdog scratch 0."""
    if _SCRATCH0 is None:
        return
    machine.mem32[_SCRATCH0] = 0 if value is None else _SCRATCH_MAGIC | (value & 0xFFFF)


def scratch_get():
    if _SCRATCH0 is None:
        return None
    v = machine.mem32[_SCRATCH0] & 0xFFFFFFFF
    if v & 0xFFF00000 != _SCRATCH_MAGIC:
        return None
    return v & 0xFFFF


def unique_id():
    try:
        return binascii.hexlify(machine.unique_id()).decode()
    except Exception:
        return "000000000000"


def mem_free():
    return gc.mem_free()


def platform():
    return _mach or sys.platform


def reset():
    if machine is not None:
        machine.reset()
    raise SystemExit(0)
