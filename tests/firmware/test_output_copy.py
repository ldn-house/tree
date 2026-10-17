"""firmware/app/_output_fallback.py must stay identical to effects/_output.py."""

import subprocess

import pytest

import devsim


def test_fallback_output_stage_is_identical():
    copy = (devsim.ROOT / "firmware" / "app" / "_output_fallback.py").read_bytes()
    ref = devsim.ROOT / "effects" / "_output.py"
    if ref.exists():
        want = ref.read_bytes()
    else:  # before the effects package lands
        r = subprocess.run(["git", "show", "origin/harness/effects-renderer:effects/_output.py"],
                           cwd=devsim.ROOT, capture_output=True)
        if r.returncode:
            pytest.skip("effects/_output.py not available")
        want = r.stdout
    assert copy == want, "copy effects/_output.py over firmware/app/_output_fallback.py"
