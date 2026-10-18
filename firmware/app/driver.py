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

import cfgcheck

from app.log import log

RESET_US = 300  # WS2811 latches after >= 280 us low

# Default waveform, in ns: (T0H, T0L, T1H, T1L). This is MicroPython's own
# neopixel/machine.bitstream 800 kHz timing, which drove this exact hardware
# (Pico GP0 -> BSS138 level shifter with 10k pull-ups -> 12 V WS2811
# bullets). The shifter's high side rises only through its pull-up (RC
# about 0.5 us with the cable), so short high pulses can vanish: don't go
# below ~400 ns for T0H. Override with LED_TIMING_NS in config.py.
DEFAULT_TIMING_NS = cfgcheck.DEFAULT_TIMING_NS
PIO_FREQ = 20_000_000
CYCLE_NS = cfgcheck.CYCLE_NS  # 50 ns at 20 MHz
TimingError = cfgcheck.ConfigError
# One validator for the app, safe mode and flash.py (firmware/cfgcheck.py).
cycles = cfgcheck.timing_cycles


def program(timing_ns=DEFAULT_TIMING_NS):
    """Assemble the WS2811 PIO program for a timing. asm_pio runs the body
    with only the PIO names as globals, so the delays come in through the
    closure (locals of this function)."""
    c1, c2, c3 = cycles(timing_ns)
    d1, d2, d3 = c1 - 1, c2 - 1, c3 - 1

    @rp2.asm_pio(
        sideset_init=rp2.PIO.OUT_LOW,
        out_shiftdir=rp2.PIO.SHIFT_LEFT,
        autopull=True,
        pull_thresh=8,
    )
    def ws2811():
        wrap_target()
        label("bitloop")
        out(x, 1).side(0)[d3]
        jmp(not_x, "do_zero").side(1)[d1]
        jmp("bitloop").side(1)[d2]
        label("do_zero")
        nop().side(0)[d2]
        wrap()

    return ws2811


class Driver:
    def __init__(self, n, pin, timing_ns=DEFAULT_TIMING_NS, sm_id=0):
        import machine

        self.n = n
        self.sm_id = sm_id
        c1, c2, c3 = cycles(timing_ns)
        self.timing_ns = (c1 * CYCLE_NS, (c2 + c3) * CYCLE_NS, (c1 + c2) * CYCLE_NS, c3 * CYCLE_NS)
        bit_ns = (c1 + c2 + c3) * CYCLE_NS
        self.write_us = n * 24 * bit_ns // 1000  # time on the wire
        self.busy_us = self.write_us + RESET_US
        self.sm = rp2.StateMachine(sm_id, program(timing_ns), freq=PIO_FREQ, sideset_base=machine.Pin(pin))
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
    timing_ns = None

    def __init__(self, n):
        self.n = n
        self.write_us = n * 24 * 1250 // 1000
        self.busy_us = self.write_us + RESET_US
        self.frames = 0

    def busy(self):
        return False

    def write(self, buf):
        self.frames += 1


def create(n, pin, timing_ns=None):
    if rp2 is None:
        return NullDriver(n)
    timing = DEFAULT_TIMING_NS if timing_ns is None else timing_ns
    try:
        cycles(timing)
    except TimingError as e:
        log("driver: bad LED_TIMING_NS %r (%s); using the default %r" % (timing, e, DEFAULT_TIMING_NS))
        timing = DEFAULT_TIMING_NS
    try:
        return Driver(n, pin, timing)
    except Exception as e:
        log("driver: PIO init failed: %r" % (e,))
        return NullDriver(n)
