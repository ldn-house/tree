"""firmware/app/_output_fallback.py must stay identical to effects/_output.py."""

import devsim


def test_fallback_output_stage_is_identical():
    copy = (devsim.ROOT / "firmware" / "app" / "_output_fallback.py").read_bytes()
    want = (devsim.ROOT / "effects" / "_output.py").read_bytes()
    assert copy == want, "copy effects/_output.py over firmware/app/_output_fallback.py"
