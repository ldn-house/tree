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


if rp2 is not None:

    @rp2.asm_pio(
        sideset_init=rp2.PIO.OUT_LOW,
        out_shiftdir=rp2.PIO.SHIFT_LEFT,
        autopull=True,
        pull_thresh=8,
    )
    def _ws2811():
        # 10 PIO cycles per bit at 8 MHz: 1 = 0.875 us high, 0 = 0.375 us.
        wrap_target()
        label("bitloop")
        out(x, 1).side(0)[2]
        jmp(not_x, "do_zero").side(1)[1]
        jmp("bitloop").side(1)[4]
        label("do_zero")
        nop().side(0)[4]
        wrap()


class Driver:
    def __init__(self, n, pin, sm_id=0):
        import machine

        self.n = n
        self.sm_id = sm_id
        self.write_us = n * 24 * BIT_NS // 1000  # time on the wire
        self.busy_us = self.write_us + RESET_US
        self.sm = rp2.StateMachine(sm_id, _ws2811, freq=8_000_000, sideset_base=machine.Pin(pin))
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
