"""tools/build_firmware.py's mpy-cross cache ($MPY_CACHE_DIR, set by the
test session) must give exactly what a fresh compile gives, skip the
compiler on a hit, and never serve output compiled from other source,
under another module name or by another mpy-cross version."""

import importlib.metadata
import os
import sys

import devsim

sys.path.insert(0, str(devsim.ROOT / "tools"))
import build_firmware  # noqa: E402


def test_cache_returns_the_fresh_compile_and_keys_on_source_name_and_compiler(tmp_path, monkeypatch):
    src = tmp_path / "m.py"
    src.write_text("X = 1\n")
    monkeypatch.delenv("MPY_CACHE_DIR", raising=False)
    build_firmware.mpy_cross(src, "app/m.py", tmp_path / "fresh.mpy")
    fresh = (tmp_path / "fresh.mpy").read_bytes()
    monkeypatch.setenv("MPY_CACHE_DIR", str(tmp_path / "cache"))
    build_firmware.mpy_cross(src, "app/m.py", tmp_path / "miss.mpy")  # fills the cache
    entries = lambda: len(os.listdir(tmp_path / "cache"))  # noqa: E731
    assert entries() == 1 and (tmp_path / "miss.mpy").read_bytes() == fresh

    # A hit copies the cached output and runs no compiler.
    def no_compiler(*a, **kw):
        raise AssertionError("mpy-cross ran on a cache hit")

    real_run = build_firmware.subprocess.run
    monkeypatch.setattr(build_firmware.subprocess, "run", no_compiler)
    build_firmware.mpy_cross(src, "app/m.py", tmp_path / "hit.mpy")
    assert (tmp_path / "hit.mpy").read_bytes() == fresh
    monkeypatch.setattr(build_firmware.subprocess, "run", real_run)

    # Another module name, changed source or another mpy-cross version compiles afresh.
    build_firmware.mpy_cross(src, "app/n.py", tmp_path / "n.mpy")
    assert entries() == 2 and (tmp_path / "n.mpy").read_bytes() != fresh
    real_version = importlib.metadata.version
    monkeypatch.setattr(importlib.metadata, "version", lambda dist: "0.0-other" if dist == "mpy-cross" else real_version(dist))
    build_firmware.mpy_cross(src, "app/m.py", tmp_path / "other.mpy")
    assert entries() == 3
    monkeypatch.setattr(importlib.metadata, "version", real_version)
    src.write_text("X = 2\n")
    build_firmware.mpy_cross(src, "app/m.py", tmp_path / "changed.mpy")
    assert entries() == 4 and (tmp_path / "changed.mpy").read_bytes() != fresh
