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


# Lowest free heap seen right after each kind of allocation-heavy step
# (effect start, MQTT publish, HTTP request), for GET / -> supervisor.heap_sites.
LOW_HEAP_WARN = 64 * 1024
heap_sites = {}


def heap_mark(site):
    free = gc.mem_free()
    low = heap_sites.get(site)
    if low is None or free < low:
        heap_sites[site] = free
        if free < LOW_HEAP_WARN:
            from app.log import log

            log("heap: only %d bytes free after %s" % (free, site))
    return free


def platform():
    return _mach or sys.platform


def reset():
    """Deliberate reset (clears the hang markers first)."""
    from app import recovery

    recovery.reset()
