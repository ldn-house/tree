#!/usr/bin/env python3
"""Plan the CI matrix for the firmware tests (.github/workflows/build.yml).

    shards.py full [N]     every test file, in N shards balanced by duration
    shards.py effects      only the tests that build the repo's effects/

Prints the matrix as JSON: {"include": [{"shard": "1/4", "files": "..."}]}.
Every test_*.py here lands in exactly one shard, so a new file is never left
out; one missing from SECONDS gets DEFAULT_S until it is measured.
"""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

# Seconds per file on a GitHub-hosted ubuntu-latest runner (pytest
# --durations, summed per file). Re-measure when a file grows a lot.
SECONDS = {
    "test_heap.py": 330,
    "test_mqtt.py": 185,
    "test_updates.py": 127,
    "test_runtime.py": 120,
    "test_review1.py": 100,
    "test_recovery_compat.py": 74,
    "test_calibration.py": 47,
    "test_real_effects.py": 43,
    "test_review2.py": 42,
    "test_resets.py": 40,
    "test_units.py": 3,
    "test_build_cache.py": 1,
    "test_output_copy.py": 1,
    "test_pio.py": 1,
}
DEFAULT_S = 60

# What an effects-only change can break: the bundle built from the repo's
# effects/ (test_real_effects, the board-sized heap campaign) and the
# firmware's copy of the output stage.
EFFECTS = ["test_real_effects.py", "test_heap.py", "test_output_copy.py"]


def plan(files, n):
    shards = [[0, []] for _ in range(n)]
    for f in sorted(files, key=lambda f: (-SECONDS.get(f, DEFAULT_S), f)):
        s = min(shards, key=lambda s: s[0])  # longest first onto the lightest
        s[0] += SECONDS.get(f, DEFAULT_S)
        s[1].append(f)
    return [sorted(fs) for _t, fs in shards if fs]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    files = sorted(p.name for p in HERE.glob("test_*.py"))
    if mode == "effects":
        groups = [[f for f in EFFECTS if f in files]]
    elif mode == "full":
        groups = plan(files, int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        sys.exit("usage: shards.py full [N] | effects")
    include = [{"shard": "%d/%d" % (i + 1, len(groups)) if mode == "full" else "effects only",
                "files": " ".join("tests/firmware/" + f for f in g)} for i, g in enumerate(groups)]
    print(json.dumps({"include": include}))


if __name__ == "__main__":
    main()
