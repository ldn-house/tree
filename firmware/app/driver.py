"""WS2811 output on a PIO state machine, fed by DMA.

The old firmware used neopixel/machine.bitstream, which bit-bangs with IRQs
off for ~15 ms per frame and stalls Wi-Fi (review M2). Here a PIO program
generates the 800 kHz waveform and a DMA channel streams the frame into the
PIO TX FIFO in the background; write() returns immediately.

The frame buffer is 3 bytes per LED already in wire order (the output stage
does colour order). DMA writes one byte at a time to the TX FIFO; the bus
replicates a narrow write across the 32-bit word, and with autopull at 8 bits
shifting left the PIO sends exactly that byte, MSB first.
"""

import time

try:
    import rp2
except ImportError:  # pragma: no cover
    rp2 = None

from app.log import log

BIT_NS = 1250
RESET_US = 300  # WS2811 latches after >= 280 us low

# Waveform: 25 PIO cycles per bit at 20 MHz (50 ns each).
#   0 bit: high T1 = 7 cycles (350 ns), low T2 + T3 = 18 (900 ns)
#   1 bit: high T1 + T2 = 16 (800 ns),   low T3 = 9 (450 ns)
# The old firmware's machine.bitstream timing (400/850 ns high for 0/1,
# known to work on this strip) is the reference. 350 ns for a 0 also sits
# inside the original WS2811 high-speed window (T0H 250 +/- 150 ns) and the
# WS2812-style window later WS2811 revisions use (T0H 250-550 ns, T1H
# 650-950 ns). tests/firmware/test_pio.py simulates this program and checks
# the pulse widths against those windows.
PIO_FREQ = 20_000_000


if rp2 is not None:

    @rp2.asm_pio(
        sideset_init=rp2.PIO.OUT_LOW,
        out_shiftdir=rp2.PIO.SHIFT_LEFT,
        autopull=True,
        pull_thresh=8,
    )
    def _ws2811():
        # asm_pio runs this body with only the PIO names as globals, so the
        # cycle counts must be locals here.
        T1 = 7
        T2 = 9
        T3 = 9
        wrap_target()
        label("bitloop")
        out(x, 1).side(0)[T3 - 1]
        jmp(not_x, "do_zero").side(1)[T1 - 1]
        jmp("bitloop").side(1)[T2 - 1]
        label("do_zero")
        nop().side(0)[T2 - 1]
        wrap()


class Driver:
    def __init__(self, n, pin, sm_id=0):
        import machine

        self.n = n
        self.sm_id = sm_id
        self.write_us = n * 24 * BIT_NS // 1000  # time on the wire
        self.busy_us = self.write_us + RESET_US
        self.sm = rp2.StateMachine(sm_id, _ws2811, freq=PIO_FREQ, sideset_base=machine.Pin(pin))
        self.sm.active(1)
        self.dma = None
        self._t = time.ticks_us() - self.busy_us
        self.frames = 0
        try:
            self.dma = rp2.DMA()
            dreq = (sm_id // 4) * 8 + (sm_id % 4)  # DREQ_PIOx_TXy
            self.ctrl = self.dma.pack_ctrl(size=0, inc_read=True, inc_write=False, treq_sel=dreq)
            self.target = self.sm
        except Exception as e:
            log("driver: no DMA (%r), using blocking FIFO writes" % (e,))
            self.dma = None
        self.mode = "pio+dma" if self.dma else "pio"

    def busy(self):
        """True until the previous frame has been sent and latched."""
        if self.dma is not None and self.dma.active():
            return True
        return time.ticks_diff(time.ticks_us(), self._t) < self.busy_us

    def write(self, buf):
        """Start sending buf (3 * n bytes, wire order). The caller must not
        modify buf until busy() is False, and must wait for busy() first."""
        self._t = time.ticks_us()
        self.frames += 1
        if self.dma is not None:
            try:
                self.dma.config(read=buf, write=self.target, count=len(buf), ctrl=self.ctrl, trigger=True)
                return
            except TypeError:
                # Older ports want the FIFO address rather than the SM object.
                self.target = _txf_address(self.sm_id)
                self.dma.config(read=buf, write=self.target, count=len(buf), ctrl=self.ctrl, trigger=True)
                return
        self.sm.put(buf, 24)


def _txf_address(sm_id):
    pio = sm_id // 4
    return 0x50200000 + pio * 0x100000 + 0x10 + 4 * (sm_id % 4)


class NullDriver:
    """No LEDs (used when PIO is unavailable)."""

    mode = "none"

    def __init__(self, n):
        self.n = n
        self.write_us = n * 24 * BIT_NS // 1000
        self.busy_us = self.write_us + RESET_US
        self.frames = 0

    def busy(self):
        return False

    def write(self, buf):
        self.frames += 1


def create(n, pin):
    if rp2 is None:
        return NullDriver(n)
    try:
        return Driver(n, pin)
    except Exception as e:
        log("driver: PIO init failed: %r" % (e,))
        return NullDriver(n)
