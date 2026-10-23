"""Tree effects. See docs/effect-api.md for the contract.

EFFECTS is ordered: it drives the Home Assistant effect list and the auto
cycle. Each name is a module in this package, imported lazily by load().
Only EFFECTS is exposed to Home Assistant, the auto cycle and the tools.

An effect module with AUTO = False is registered (Home Assistant can select
it, takeovers can show it) but the auto cycle skips it.

DISABLED lists modules kept in the package but not registered, so they can
be re-enabled by moving the name back into EFFECTS. Nothing loads them.
"""

EFFECTS = [
    "rainbow_cycle",
    "christmas",
    "candy_cane",
    "twinkle_multi",
    "warm_white",
    "plane_sweep",
    "radial_burst",
    "spiral_3d",
    "lighthouse",
    "raindrops",
    "snowfall",
    "pong",
    "level_up",
    "beacon",
    "tetris",
    "portal",
    "fire",
    "lightning",
    "storm",
    "solid",
    "f1_start",
    "f1_yellow",
    "f1_red",
    "f1_chequered",
    "f1_winner",
]

DISABLED = [
    "sparkle",  # rapid white flashing; disabled 2026-10-18 at the user's request
    "space_invaders",  # disabled 2026-10-21 at the user's request
]


def load(name):
    """Import and return the effect module for name."""
    mod = __import__("effects." + name)
    return getattr(mod, name)


def defaults(module):
    """Default params dict for an effect module."""
    params = {}
    for k, spec in getattr(module, "PARAMS", {}).items():
        params[k] = spec[0]
    return params


class Ctx:
    """The ctx passed to Effect(ctx). The runtime may use any object with
    these attributes."""

    def __init__(self, n, coords, params, random):
        self.n = n
        self.coords = coords
        self.params = params
        self.random = random
