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

_mach = getattr(sys.implementation, "_machine", "")


def scratch_set(value):
    """Watchdog scratch 0 (bootmgr's, or app.recovery's on an older recovery layer)."""
    from app import recovery

    recovery.scratch_set(value)


def scratch_get():
    from app import recovery

    return recovery.scratch_get()


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
    """Deliberate reset (clears the hang markers first)."""
    from app import recovery

    recovery.reset()
