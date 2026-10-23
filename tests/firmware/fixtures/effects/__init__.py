"""Stand-in effects package for firmware tests, shaped like the real one
(docs/effect-api.md). The firmware is tested against this until the
harness thread's effects/ package lands."""

EFFECTS = ["solid", "chase", "height", "tint"]


def load(name):
    mod = __import__("effects." + name)
    return getattr(mod, name)


class Ctx:
    def __init__(self, n, coords, params, random):
        self.n = n
        self.coords = coords
        self.params = params
        self.random = random
