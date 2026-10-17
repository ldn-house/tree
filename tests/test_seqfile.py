"""Test vectors for the .tcal format (docs/calibration-protocol.md, section 4)."""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from calib import seqfile  # noqa: E402
from calib.sequence import SequenceParams, build  # noqa: E402

DATA = ROOT / "tests" / "data"


def expanded_hash(data: bytes) -> str:
    n, _loops, frames = seqfile.decode(data)
    h = hashlib.sha256()
    for f in frames:
        h.update(bytes(f.rgb(n)))
    return h.hexdigest()


def test_default_vector_matches_hash():
    data = (DATA / "calib-default.tcal").read_bytes()
    expected = (DATA / "calib-default.expanded.sha256").read_text().split()[0]
    assert expanded_hash(data) == expected


def test_default_vector_is_current_default():
    assert (DATA / "calib-default.tcal").read_bytes() == (ROOT / "sequences" / "default.tcal").read_bytes()


def test_roundtrip_matches_plan():
    for params in (SequenceParams(), SequenceParams(colour=True), SequenceParams(codebook="gray")):
        plan = build(params)
        n, _loops, frames = seqfile.decode(plan.to_tcal())
        assert n == 500 and len(frames) == plan.n_frames
        for i, f in enumerate(frames):
            assert bytes(f.rgb(n)) == plan.rgb()[i].astype(np.uint8).tobytes()
            assert f.duration_ms == plan.duration_ms[i]


def test_frames_stay_under_power_budget():
    for params in (SequenceParams(), SequenceParams(colour=True)):
        assert build(params).frame_power().max() <= params.power_budget


def test_rejects_corruption():
    data = bytearray((DATA / "calib-default.tcal").read_bytes())
    data[100] ^= 1
    try:
        seqfile.decode(bytes(data))
    except ValueError as e:
        assert "crc" in str(e)
    else:
        raise AssertionError("corrupt file accepted")


def test_bit_order_lsb_first():
    plane = seqfile.pack_plane([1, 0, 0, 0, 0, 0, 0, 0, 0, 1], 10)
    assert plane == bytes([0b00000001, 0b00000010])
