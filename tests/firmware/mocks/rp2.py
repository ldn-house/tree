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


class _Instr:
    def __init__(self, op, *args):
        self.op = op
        self.args = args
        self.sideset = None
        self.delay = 0

    def side(self, v):
        self.sideset = v
        return self

    def __getitem__(self, d):
        if not 0 <= d <= 15:  # 5 delay/side-set bits, 1 used for side-set
            raise ValueError("PIO delay %d out of range" % d)
        self.delay = d
        return self


class Program:
    def __init__(self, config):
        self.config = config
        self.instrs = []
        self.labels = {}
        self.wrap_target = 0
        self.wrap = None


def asm_pio(**kw):
    """Assemble like the real rp2.asm_pio: run the body with ONLY the PIO
    instruction names as globals (so module-level names are not visible),
    recording the program for simulate()."""

    def deco(fn):
        prog = Program(kw)

        def emit(op):
            def f(*args):
                ins = _Instr(op, *args)
                prog.instrs.append(ins)
                return ins

            return f

        def label(name):
            prog.labels[name] = len(prog.instrs)

        def wrap_target():
            prog.wrap_target = len(prog.instrs)

        def wrap():
            prog.wrap = len(prog.instrs) - 1

        names = {"out": emit("out"), "jmp": emit("jmp"), "nop": emit("nop"), "set": emit("set"),
                 "label": label, "wrap_target": wrap_target, "wrap": wrap,
                 "x": "x", "y": "y", "not_x": "!x", "not_y": "!y", "pins": "pins", "null": "null"}
        g = fn.__globals__
        saved = dict(g)
        g.clear()
        g.update(names)
        try:
            fn()
        finally:
            g.clear()
            g.update(saved)
        return prog

    return deco


def simulate(prog, data, cycles_limit=10000000):
    """Run a recorded WS2811-style program over data (bytes, as DMA would
    write them: each byte replicated across the 32-bit FIFO word). Returns
    the pin waveform as a list of (level, cycles) runs until the FIFO is
    empty. Supports out/jmp/nop with side-set, autopull and wrap."""
    thresh = prog.config.get("pull_thresh", 32)
    fifo = [(b << 24) | (b << 16) | (b << 8) | b for b in data]
    osr = 0
    count = thresh  # empty: autopull before the first out
    x = 0
    pc = prog.wrap_target
    pin = 0
    runs = []
    total = 0

    def level(v, n):
        if runs and runs[-1][0] == v:
            runs[-1] = (v, runs[-1][1] + n)
        else:
            runs.append((v, n))

    while total < cycles_limit:
        ins = prog.instrs[pc]
        if ins.op == "out" and count >= thresh:
            if not fifo:
                break
            osr = fifo.pop(0)
            count = 0
        if ins.sideset is not None:
            pin = ins.sideset
        nxt = pc + 1
        if ins.op == "out":
            n = ins.args[1]
            val = (osr >> (32 - n)) & ((1 << n) - 1)  # shift left: MSB first
            osr = (osr << n) & 0xFFFFFFFF
            count += n
            if ins.args[0] == "x":
                x = val
        elif ins.op == "jmp":
            if len(ins.args) == 1:
                nxt = prog.labels[ins.args[0]]
            elif ins.args[0] == "!x" and x == 0:
                nxt = prog.labels[ins.args[1]]
        level(pin, 1 + ins.delay)
        total += 1 + ins.delay
        if pc == prog.wrap and nxt == pc + 1:
            nxt = prog.wrap_target
        pc = nxt
    return runs


class StateMachine:
    def __init__(self, id, prog=None, freq=None, sideset_base=None, **kw):
        if not isinstance(prog, Program):
            raise TypeError("StateMachine needs an assembled program")
        self.id = id
        self.prog = prog
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
