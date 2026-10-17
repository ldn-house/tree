"""machine mock for running the firmware on the MicroPython unix port.

Driven by tests/firmware/devsim.py through environment variables:
    DEVSIM_WDT      file touched on every WDT.feed() (the harness resets the
                    "board" if it goes stale for longer than the timeout)
    DEVSIM_RESET    reset cause for this boot: pwron | wdt
"""

import os
import time

PWRON_RESET = 1
HARD_RESET = 2
WDT_RESET = 3

_wdt_file = os.getenv("DEVSIM_WDT")


def _exit(code):
    try:
        import ffi

        for lib in ("libc.so.6", "libc.dylib", "libSystem.dylib"):
            try:
                ffi.open(lib).func("v", "_exit", "i")(code)
            except OSError:
                pass
    except ImportError:
        pass
    raise SystemExit(code)


def reset():
    """Like the hardware: the process ends here, no unwinding."""
    print("[machine] reset()")
    _exit(42)


def soft_reset():
    _exit(43)


def reset_cause():
    return WDT_RESET if os.getenv("DEVSIM_RESET") == "wdt" else PWRON_RESET


def unique_id():
    return b"\xde\xad\xbe\xef\x00\x01"


def freq(*a):
    return 150000000


class WDT:
    def __init__(self, id=0, timeout=5000):
        self.timeout = timeout
        if _wdt_file:
            with open(_wdt_file + ".timeout", "w") as f:
                f.write(str(timeout))
        self.feed()

    def feed(self):
        if _wdt_file:
            with open(_wdt_file, "w") as f:
                f.write(str(time.time()))


class Pin:
    OUT = 1
    IN = 0

    def __init__(self, id, mode=None, value=None):
        self.id = id

    def value(self, v=None):
        return 0


class Timer:
    PERIODIC = 1
    ONE_SHOT = 0

    def __init__(self, *a, **kw):
        # No soft-timer IRQs on the unix port. The harness only judges the
        # watchdog once the firmware has fed it at least once.
        pass

    def deinit(self):
        pass


def bitstream(pin, encoding, timing, data):
    pass
