"""The runtime output stage (docs/effect-api.md, "Output stage").

The reference implementation is effects/_output.py, shared with the renderer
and viewer so the tree and the previews cap and dim identically. The copy
below is only a safety net for a bundle without effects/: the power cap must
apply whatever else is broken.
"""

try:
    from effects._output import Output, budget_for
except Exception:  # pragma: no cover - exercised by tests without effects/
    _Q = 12

    def budget_for(n, percent=13):
        return n * 765 * percent // 100

    class Output:
        def __init__(self, n, budget=None, gamma=None, order="RGB"):
            self.n = n
            self.budget = budget_for(n) if budget is None else int(budget)
            self.order = bytes(["RGB".index(c) for c in order])
            self.lut = bytearray(256)
            self.gamma = None
            if gamma is not None and gamma != 1.0:
                g = bytearray(256)
                for v in range(256):
                    g[v] = int(255 * (v / 255) ** gamma + 0.5)
                self.gamma = g
            self.raw_total = 0
            self.capped = False
            self.factor = 1 << _Q

        def process(self, src, dst, brightness):
            total = sum(src)
            self.raw_total = total
            if total > self.budget:
                f = (self.budget << _Q) // total
                self.capped = True
            else:
                f = 1 << _Q
                self.capped = False
            self.factor = f
            s = f * brightness // 255
            lut = self.lut
            g = self.gamma
            for v in range(256):
                lut[v] = (v * s) >> _Q if g is None else g[(v * s) >> _Q]
            o0, o1, o2 = self.order[0], self.order[1], self.order[2]
            for k in range(0, 3 * self.n, 3):
                dst[k] = lut[src[k + o0]]
                dst[k + 1] = lut[src[k + o1]]
                dst[k + 2] = lut[src[k + o2]]


FACTOR_ONE = 4096


def create(n, percent, gamma, order):
    return Output(n, budget_for(n, percent), gamma, order)
