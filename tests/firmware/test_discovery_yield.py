"""Discovery must cooperate even when MQTT accepts every send immediately."""
import subprocess
from pathlib import Path

import pytest
import devsim


@pytest.mark.parametrize("cost_ms", [0, 20])
def test_discovery_yields_and_preserves_retained_behavior(tmp_path, cost_ms):
    script = Path(__file__).parent / "mp" / "discovery_yield.py"
    result = subprocess.run(
        [devsim.MICROPYTHON, str(script), str(devsim.ROOT / "firmware"), str(tmp_path), str(cost_ms)],
        capture_output=True, text=True, timeout=45,
    )
    print(result.stdout)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "discovery behavior verified" in result.stdout
