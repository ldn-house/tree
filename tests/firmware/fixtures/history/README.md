# Firmware at earlier commits

Snapshots of `firmware/` (recovery layer and app) at commits the tests compare against. They were reviewed or deployed on the PR #21 branch, which was squash-merged, so the commits aren't reachable from `main` and a CI checkout can't `git archive` them.

| Directory | Commit | Used by |
|---|---|---|
| `52cda75/` | 52cda75 Add review round 2 regression tests and docs | `test_recovery_compat.py`: the recovery layer and app first installed on the tree |
| `57c4ce1/` | 57c4ce1 Fix flash.py header merge and skip MicroPython-only tests in pytest (reviewed in round 1) | `test_units.py`: round-1 reproductions must fail here |
| `1d0b869/` | 1d0b869 State the power cap in amps (reviewed in round 2) | `test_units.py`: round-2 reproductions must fail here |
| `183d9b8/` | 183d9b8 Test that the retained running effect tracks the latest (reviewed in round 3) | `test_units.py`: round-3 reproductions must fail here |

Each holds `firmware/` exactly as `git archive <commit> firmware` produced it, from a checkout that still had the PR branch. Don't edit them: they are the old code under test.
