"""Where was the event loop when the watchdog fired?

A hang that blocks the event loop for 8 s resets the board before anything
can be logged, and the RAM log doesn't survive. So each task marks what it
is doing in watchdog scratch 1, which survives a watchdog reset (scratch 0
is the running effect, see engine; 4-7 belong to the boot ROM):

    0x7EE1 << 16 | phase << 8 | timer armed (0x80) | stalled seconds (0-127)

A 1 s soft timer adds how long the loop has been stalled. Soft-timer
callbacks run between bytecodes, so a stall counted there means Python code
was running without yielding; a hang inside a native call (a socket, the
flash, the Wi-Fi driver) stops the callbacks too and leaves 0. After a
watchdog reset with no recorded reason, the next boot reports both
(GET / -> boot.hang, the log, and HA's "Last reset cause").
"""

import time

from app import recovery

try:
    import machine
except ImportError:  # pragma: no cover
    machine = None

IDLE = 0
RENDER = 1
OUTPUT = 2
EFFECT_START = 3
MQTT = 4
HA = 5
HTTP = 6
STATE_SAVE = 7
COORDS = 8
UPDATE = 9
CALIB = 10
GC = 11
RESERVE = 12
WIFI = 13
SUPERVISOR = 14
COLOUR = 15
NAMES = ("idle", "effect render", "output stage", "effect start", "mqtt", "home assistant", "http request",
         "state save", "coordinates", "update", "calibration", "gc", "reserve", "wifi", "supervisor",
         "colour floor")

_MAGIC = 0x7EE10000
_SCRATCH1 = 0x10

_base = recovery._watchdog_base()
_addr = None if _base is None else _base + _SCRATCH1
phase = IDLE
stall = 0
armed = 0
tick = time.ticks_ms()  # the supervisor's last turn of the event loop
_timer = None


def _write():
    if _addr is not None:
        machine.mem32[_addr] = _MAGIC | phase << 8 | armed | stall


def at(p):
    global phase
    if p != phase:
        phase = p
        _write()


def alive(now):
    """Called by the supervisor every turn of the loop."""
    global tick, stall
    tick = now
    if stall:
        stall = 0
        _write()


def _check(_t=None):
    global stall
    s = time.ticks_diff(time.ticks_ms(), tick) // 1000
    if s >= 2 and s != stall:
        stall = min(s, 127)
        _write()


def start():
    global _timer, armed
    if machine is None or _addr is None:
        return
    try:
        _timer = machine.Timer(period=1000, mode=machine.Timer.PERIODIC, callback=_check)
        armed = 0 if getattr(_timer, "DEVSIM_NO_IRQ", False) else 0x80
    except Exception:
        _timer = None
    _write()


def take():
    """The breadcrumb left by the previous boot, cleared: {"phase", "stalled_s"} or None."""
    if _addr is None:
        return None
    v = machine.mem32[_addr] & 0xFFFFFFFF
    machine.mem32[_addr] = 0
    if v & 0xFFFF0000 != _MAGIC:
        return None
    p = (v >> 8) & 0xFF
    return {"phase": NAMES[p] if p < len(NAMES) else str(p), "stalled_s": v & 0x7F, "timer": bool(v & 0x80)}


def clear():
    if _addr is not None:
        machine.mem32[_addr] = 0


def describe(h):
    if not h:
        return None
    if h["stalled_s"]:
        return "hung in %s (Python ran %d s without yielding)" % (h["phase"], h["stalled_s"])
    if h["timer"]:
        return "hung in %s (blocked in native code)" % h["phase"]
    return "hung in %s" % h["phase"]
