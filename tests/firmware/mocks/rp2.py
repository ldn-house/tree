"""rp2 mock: PIO state machines and DMA, enough for app/driver.py.

The DMA mock "sends" a frame by copying it and staying active for the time
the real transfer would take (1.25 us per bit), so frame pacing and the
calibration player's latch timing behave like the hardware.
"""

import time


class PIO:
    OUT_LOW = 0
    OUT_HIGH = 1
    SHIFT_LEFT = 0
    SHIFT_RIGHT = 1


def asm_pio(**kw):
    def deco(fn):
        return fn

    return deco


class StateMachine:
    def __init__(self, id, prog=None, freq=None, sideset_base=None, **kw):
        self.id = id
        self.freq = freq
        self.puts = 0

    def active(self, v=None):
        return 1

    def put(self, buf, shift=0):
        self.puts += 1


frames = 0
last = None


class DMA:
    def __init__(self):
        self._until = time.ticks_us()

    def pack_ctrl(self, **kw):
        return kw

    def config(self, read=None, write=None, count=0, ctrl=None, trigger=False):
        global frames, last
        frames += 1
        last = bytes(read[:count])
        self._until = time.ticks_add(time.ticks_us(), count * 10)  # 8 bits x 1.25 us

    def active(self, v=None):
        return time.ticks_diff(self._until, time.ticks_us()) > 0

    def close(self):
        pass
