"""Standard calibration sequences: sync preamble, references and code frames.

The firmware plays any .tcal file verbatim; this module decides what goes in
it. The decoder (calib.video) rebuilds the same plan from the parameters, so
the layout below is the contract between the two, not the firmware.

Layout (one loop):

    lead      all off, lead_ms (gives the camera time to settle; loop marker)
    sync      13-symbol Barker code of all-on / all-off frames
    off, on   reference frames
    r, g, b   colour training frames (colour mode only)
    code      codeword bits, as pattern (+ complement when differential);
              colour mode carries three bits per frame, one per channel
    tail      Barker code again, so drift can be measured across the loop
"""

from dataclasses import dataclass, field

import numpy as np

from . import seqfile
from .codes import Codebook, make_codebook

BARKER13 = np.array([1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1], np.uint8)

# Raw RGB sum per frame the firmware's power cap allows, as reported by
# GET /calibrate (power_budget). The rewrite firmware (#21) caps a single 12 V
# feed at 1.5 A with 8 mA per fully driven WS2811 channel: 1500 / 8 * 255.
# Frames are built to stay under it so the cap never rescales a calibration
# frame. Prefer the device's value (calibrate.py sequence --host).
DEFAULT_POWER_BUDGET = 47812


@dataclass
class SequenceParams:
    codebook: str = "hamming"
    differential: bool = True
    colour: bool = False
    symbol_ms: int = 100
    lead_ms: int = 1000
    n_leds: int = 500
    power_budget: int = DEFAULT_POWER_BUDGET
    max_level: int = 255
    sync: bool = True

    def key(self) -> str:
        return (f"{self.codebook}{'-diff' if self.differential else ''}"
                f"{'-rgb' if self.colour else ''}-{self.symbol_ms}ms")


@dataclass
class Plan:
    params: SequenceParams
    codebook: Codebook
    roles: list[str]                # per frame
    bits: list[tuple]               # per frame: ((bit, polarity) per channel) or ()
    on: np.ndarray                  # (F, 3, n) bool: channel lit
    level: np.ndarray               # (F, 3) uint8
    duration_ms: np.ndarray         # (F,) int
    extra: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return len(self.roles)

    def starts_ms(self) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(self.duration_ms)[:-1]])

    @property
    def loop_ms(self) -> int:
        return int(self.duration_ms.sum())

    def rgb(self) -> np.ndarray:
        """(F, n, 3) float LED drive levels 0-255."""
        return (self.on.transpose(0, 2, 1) * self.level[:, None, :]).astype(np.float32)

    def frame_power(self) -> np.ndarray:
        return self.rgb().sum(axis=(1, 2))

    def index(self, role: str) -> list[int]:
        return [i for i, r in enumerate(self.roles) if r == role]

    def to_frames(self) -> list[seqfile.Frame]:
        out = []
        for f in range(self.n_frames):
            lit = self.on[f]
            mono = all((lit[c] == lit[0]).all() or self.level[f, c] == 0 for c in range(3))
            if mono:
                ref = next((c for c in range(3) if self.level[f, c]), 0)
                planes = (seqfile.pack_plane(lit[ref], self.params.n_leds),)
            else:
                planes = tuple(seqfile.pack_plane(lit[c], self.params.n_leds) for c in range(3))
            out.append(seqfile.Frame(int(self.duration_ms[f]), tuple(int(v) for v in self.level[f]), planes))
        return out

    def to_tcal(self, loops: int = 0) -> bytes:
        return seqfile.encode(self.to_frames(), self.params.n_leds, loops)


def build(params: SequenceParams) -> Plan:
    n = params.n_leds
    cb = make_codebook(params.codebook, n)
    roles, bits, on = [], [], []

    def add(role, lit, b=()):
        roles.append(role)
        bits.append(b)
        on.append(np.asarray(lit, bool).reshape(3, n))

    all_on = np.ones((3, n), bool)
    off = np.zeros((3, n), bool)
    add("lead", off)
    if params.sync:
        for s in BARKER13:
            add("sync", all_on if s else off)
    add("off", off)
    add("on", all_on)
    if params.colour:
        for c, name in enumerate("rgb"):
            lit = np.zeros((3, n), bool)
            lit[c] = True
            add(name, lit)
    words = cb.words.T.astype(bool)  # (bits, n)
    polarities = (1, 0) if params.differential else (1,)
    if params.colour:
        groups = [list(range(i, min(i + 3, cb.n_bits))) for i in range(0, cb.n_bits, 3)]
        for g in groups:
            for pol in polarities:
                lit = np.zeros((3, n), bool)
                for c, b in enumerate(g):
                    lit[c] = words[b] if pol else ~words[b]
                add("code", lit, tuple((b, pol) for b in g))
    else:
        for b in range(cb.n_bits):
            for pol in polarities:
                w = words[b] if pol else ~words[b]
                add("code", np.repeat(w[None], 3, 0), ((b, pol),))
    if params.sync:
        for s in BARKER13:
            add("tail", all_on if s else off)

    on = np.stack(on)
    roles_arr = np.array(roles)
    # Code and colour-training frames share one level, so the decoder can unmix
    # and compare them directly; it is the largest that keeps the brightest of
    # them under the power budget. Sync and reference frames light every LED,
    # so they get whatever level fits (they only carry timing and a mask).
    sums = on.sum(axis=(1, 2))
    shared = np.isin(roles_arr, ["code", "r", "g", "b"])
    code_lvl = int(min(params.max_level, params.power_budget // max(sums[shared].max(), 1)))
    lvl = np.minimum(code_lvl, params.power_budget // np.maximum(sums, 1))
    lvl[shared] = code_lvl
    level = np.repeat(lvl[:, None], 3, 1).astype(np.uint8)
    dur = np.full(len(roles), params.symbol_ms, int)
    dur[roles_arr == "lead"] = params.lead_ms
    return Plan(params, cb, roles, bits, on, level, dur)


def sequential_plan(n_leds: int = 500, symbol_ms: int = 100, level: int = 255,
                    power_budget: int = DEFAULT_POWER_BUDGET) -> Plan:
    """One LED at a time, the video equivalent of capture.py's sequential mode."""
    params = SequenceParams(codebook="binary", differential=False, symbol_ms=symbol_ms,
                            n_leds=n_leds, sync=True, power_budget=power_budget)
    cb = make_codebook("binary", n_leds)
    roles, bits, on = [], [], []
    off = np.zeros((3, n_leds), bool)
    all_on = np.ones((3, n_leds), bool)

    def add(role, lit, b=()):
        roles.append(role)
        bits.append(b)
        on.append(lit)

    add("lead", off)
    for s in BARKER13:
        add("sync", all_on if s else off)
    add("off", off)
    add("on", all_on)
    for i in range(n_leds):
        lit = np.zeros((3, n_leds), bool)
        lit[:, i] = True
        add("single", lit, (i,))
    for s in BARKER13:
        add("tail", all_on if s else off)
    on = np.stack(on)
    roles_arr = np.array(roles)
    level_arr = np.zeros((len(roles), 3), np.uint8)
    full_lvl = min(level, power_budget // (3 * n_leds))
    level = min(level, power_budget // 3)
    level_arr[:] = full_lvl
    level_arr[roles_arr == "single"] = level
    dur = np.full(len(roles), symbol_ms, int)
    dur[roles_arr == "lead"] = 1000
    return Plan(params, cb, roles, bits, on, level_arr, dur)
