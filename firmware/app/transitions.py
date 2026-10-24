"""Display-only transitions. Integer maths, no allocation in step/apply."""
import time

STYLES = ("none", "fade", "wipe", "dissolve")


class Ramp:
    def __init__(self, value=0):
        self.value = value
        self.start = value
        self.target = value
        self.at = 0
        self.ms = 0
        self.pending = False

    def set(self, target, now, ms=None):
        # Start from the last displayed value, including a reversal.
        self.start = self.value
        self.target = target
        self.at = now
        self.ms = max(1, abs(target - self.value) * 1000 // 255) if ms is None else ms
        if target == self.value:
            self.ms = 0
        self.pending = bool(self.ms)
        if not self.ms:
            self.value = target

    def step(self, now):
        if self.pending:
            self.at = now
            self.pending = False
        dt = max(0, time.ticks_diff(now, self.at))
        if dt >= self.ms:
            self.value = self.target
        else:
            # Zero is a discontinuity: never display below the effective floor.
            a = max(1, self.start)
            b = max(1, self.target)
            self.value = a + (b - a) * dt // self.ms
        return self.value


class Blend:
    def __init__(self, n):
        self.old = bytearray(3 * n)
        self.order = bytearray(n)
        self.order_style = "wipe"
        self.points = None
        self.at = 0
        self.ms = 0
        self.style = "none"
        self.coords(None)

    def _order(self, style):
        n = len(self.order)
        points = self.points
        x = 17
        for i in range(n):
            if style == "dissolve":
                x = (109 * x + 89) & 255
                self.order[i] = x
            else:
                p = points[i] if points is not None and i < len(points) else None
                self.order[i] = min(255, max(0, int(p[1] * 255))) if p is not None else i * 256 // n
        self.order_style = style

    def coords(self, coords):
        self.points = coords
        self._order(self.order_style)

    def begin(self, wire, style, ms, now):
        if style in ("wipe", "dissolve") and style != self.order_style:
            self._order(style)
        self.old[:] = wire
        self.style = style
        self.ms = ms if style != "none" else 0
        self.at = now

    def apply(self, wire, now, budget):
        if not self.ms:
            return
        dt = max(0, time.ticks_diff(now, self.at))
        if dt >= self.ms:
            self.ms = 0
            return
        q = dt * 256 // self.ms
        order = self.order
        _mix(self.old, wire, order, len(order), q, int(self.style == "fade"), budget)

_KERNEL = """
@micropython.viper
def _mix_v(old: ptr8, wire: ptr8, order: ptr8, n: int, q: int, fade: int, budget: int):
    i = 0
    total = 0
    while i < n:
        w = q
        if not fade:
            w = (q * 288 // 256 - order[i]) * 8
            if w < 0:
                w = 0
            if w > 256:
                w = 256
        k = i * 3
        end = k + 3
        while k < end:
            v = (old[k] * (256 - w) + wire[k] * w) >> 8
            wire[k] = v
            total += v
            k += 1
        i += 1
    if total > budget:
        k = 0
        while k < n * 3:
            wire[k] = wire[k] * budget // total
            k += 1

"""

# Like the output stage: compile native loops at import, leaving mpy-cross
# bundles portable. The bytecode fallback has identical integer rounding.
NATIVE = False
try:
    import micropython
    _ns = {"micropython": micropython}
    exec(_KERNEL, _ns)
    _mix = _ns["_mix_v"]
    NATIVE = True
except Exception:
    _ns = {}
    exec(_KERNEL.replace("@micropython.viper\n", "").replace(": ptr8", "").replace(": int", ""), _ns)
    _mix = _ns["_mix_v"]

del _ns
del _KERNEL
