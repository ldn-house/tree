#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpy-cross==1.27.0.post2"]
# ///
"""Build the firmware bundle that the device installs (see firmware/bootmgr.py).

    tools/build_firmware.py                      # dist/tree-firmware.bin + dist/manifest.json
    tools/build_firmware.py --key KEY            # sign with an HMAC-SHA256 key
    tools/build_firmware.py --source             # ship .py instead of .mpy

Contents of a slot:
    app/*.mpy           firmware/app (compiled with mpy-cross)
    effects/*.mpy       effects/ package (compiled; the repo's effects/ unless
                        --effects says otherwise), minus modules in DISABLED
    data/effects.json   effect metadata (NAME, DURATION_MS, PARAMS, NEEDS_COORDS)
    data/coords_compact.txt   legacy coordinates, if present
    data/calib_default.tcal   standard calibration sequence, if present

The recovery layer (firmware/boot.py, main.py, bootmgr.py, safemode.py,
cfgcheck.py) is not in the bundle: it is only ever installed over USB. The
manifest's "bootmgr" is the oldest recovery API the app runs on
(RECOVERY_API in firmware/app/__init__.py); an older recovery layer refuses
the bundle before staging anything.

The signing key comes from --key or $FIRMWARE_SIGNING_KEY. The device must
have the same string as UPDATE_KEY in config.py.
"""

import argparse
import datetime
import hashlib
import hmac
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MAGIC = b"TREEFW1\n"


def git(*args, default=""):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return default


def version_info():
    commit = os.environ.get("GITHUB_SHA") or git("rev-parse", "HEAD", default="unknown")
    ref = os.environ.get("GITHUB_REF_NAME") or git("rev-parse", "--abbrev-ref", "HEAD", default="local")
    if os.environ.get("GITHUB_REF_TYPE") == "tag":
        version = ref
    else:
        version = commit[:7]
        if not os.environ.get("GITHUB_SHA") and git("status", "--porcelain", "--untracked-files=no"):
            version += "-dirty"
    now = datetime.datetime.now(datetime.timezone.utc)
    # "generation" orders bundles for the device's anti-replay floor: it is
    # signed with the manifest, and the device refuses a bundle older than one
    # it has already confirmed healthy. Build time works for CI and local
    # builds alike; equal generations are accepted.
    return {"version": version, "commit": commit, "ref": ref, "built": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "generation": int(now.timestamp())}


def recovery_api(app_dir: Path):
    """The oldest recovery-layer API the app runs on (RECOVERY_API in
    app/__init__.py), not the newest bootmgr: the manifest's "bootmgr" is
    what the device's recovery layer must provide, and it refuses more."""
    m = re.search(r"^RECOVERY_API = (\d+)\b", (app_dir / "__init__.py").read_text(), re.M)
    if not m:
        sys.exit(f"{app_dir}/__init__.py does not declare RECOVERY_API")
    return int(m.group(1))


def mpy_cross(src: Path, name: str, out: Path):
    # $MPY_CACHE_DIR (the tests set it) reuses output compiled from the same
    # source, module name and mpy-cross version; builds without it are
    # unchanged.
    cache = os.environ.get("MPY_CACHE_DIR")
    if cache:
        from importlib.metadata import version

        h = hashlib.sha256(b"\0".join([version("mpy-cross").encode(), b"-s", name.encode(), src.read_bytes()]))
        hit = Path(cache) / (h.hexdigest() + ".mpy")
        if hit.exists():
            shutil.copyfile(hit, out)
            return
    cmd = [sys.executable, "-m", "mpy_cross", "-s", name, "-o", str(out), str(src)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit(f"mpy-cross failed for {name}:\n{r.stderr or r.stdout}")
    if cache:
        hit.parent.mkdir(parents=True, exist_ok=True)
        tmp = hit.with_suffix(".%d.tmp" % os.getpid())
        shutil.copyfile(out, tmp)
        os.replace(tmp, hit)  # atomic: concurrent builds may share the cache


def effect_meta(effects_dir: Path):
    """Import the effects package with CPython and read each effect's
    metadata, so the device need not import every effect to list params.
    Returns (metadata, names of the modules in DISABLED)."""
    sys.path.insert(0, str(effects_dir.parent))
    try:
        import importlib

        pkg = importlib.import_module(effects_dir.name)
        meta = {}
        for name in pkg.EFFECTS:
            mod = importlib.import_module(f"{effects_dir.name}.{name}")
            params = {k: list(v[:3]) for k, v in getattr(mod, "PARAMS", {}).items()}
            meta[name] = {
                "duration_ms": int(getattr(mod, "DURATION_MS", 20000)),
                "params": params,
                "needs_coords": bool(getattr(mod, "NEEDS_COORDS", False)),
            }
        return meta, set(getattr(pkg, "DISABLED", ()))
    finally:
        sys.path.pop(0)


def collect(effects_dir: Path | None, source: bool, work: Path, app_dir: Path):
    """Return a list of (slot path, bytes)."""
    files = []

    def add_py(src: Path, dest: str):
        if source:
            data = src.read_bytes()
            compile(data, dest, "exec")
            files.append((dest, data))
        else:
            out = work / (dest.replace("/", "__") + ".mpy")
            mpy_cross(src, dest, out)
            files.append((dest[:-3] + ".mpy", out.read_bytes()))

    for src in sorted(app_dir.glob("*.py")):
        add_py(src, f"app/{src.name}")
    if effects_dir and (effects_dir / "__init__.py").exists():
        meta, disabled = effect_meta(effects_dir)
        for src in sorted(effects_dir.glob("*.py")):
            if src.stem in disabled:  # kept in the repo, not registered: not shipped
                continue
            add_py(src, f"effects/{src.name}")
        files.append(("data/effects.json", json.dumps(meta, sort_keys=True).encode()))
    else:
        print("warning: no effects/ package; the firmware will run its built-in fallback only", file=sys.stderr)
    coords = ROOT / "coordinates" / "coords_compact.txt"
    if coords.exists():
        files.append(("data/coords_compact.txt", coords.read_bytes()))
    seq = ROOT / "sequences" / "default.tcal"
    if seq.exists():
        files.append(("data/calib_default.tcal", seq.read_bytes()))
    return files


def bundle(files, info, key: str | None, api: int):
    manifest = {
        "format": 1,
        **info,
        "bootmgr": api,
        "files": [{"path": p, "size": len(d), "sha256": hashlib.sha256(d).hexdigest()} for p, d in files],
    }
    mbytes = json.dumps(manifest, separators=(",", ":")).encode()
    sig = hmac.new(key.encode(), mbytes, hashlib.sha256).digest() if key else b""
    out = bytearray(MAGIC)
    out += len(mbytes).to_bytes(4, "big") + mbytes
    out += len(sig).to_bytes(4, "big") + sig
    for _p, d in files:
        out += d
    return bytes(out), mbytes, manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "dist"), help="output directory (default: dist/)")
    ap.add_argument("--effects", default=str(ROOT / "effects"), help="effects package directory")
    ap.add_argument("--key", default=os.environ.get("FIRMWARE_SIGNING_KEY") or None, help="HMAC signing key")
    ap.add_argument("--source", action="store_true", help="bundle .py sources instead of .mpy")
    ap.add_argument("--version", help="override the version string")
    ap.add_argument("--generation", type=int, help="override the generation (anti-replay order)")
    ap.add_argument("--app", default=str(ROOT / "firmware" / "app"), help=argparse.SUPPRESS)  # tests: patched app
    ap.add_argument("--recovery-api", type=int, help=argparse.SUPPRESS)  # tests: override RECOVERY_API
    args = ap.parse_args()

    info = version_info()
    if args.version:
        info["version"] = args.version
    if args.generation is not None:
        info["generation"] = args.generation
    with tempfile.TemporaryDirectory() as tmp:
        files = collect(Path(args.effects), args.source, Path(tmp), Path(args.app))
    api = args.recovery_api if args.recovery_api is not None else recovery_api(Path(args.app))
    data, mbytes, manifest = bundle(files, info, args.key, api)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tree-firmware.bin").write_bytes(data)
    (out / "manifest.json").write_bytes(mbytes)
    total = sum(f["size"] for f in manifest["files"])
    print(f"{out / 'tree-firmware.bin'}: version {info['version']}, {len(files)} files, "
          f"{total} bytes, {'signed' if args.key else 'unsigned'}, needs recovery API {api}")


if __name__ == "__main__":
    main()
