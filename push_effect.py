#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Push effect files to the tree over Wi-Fi, without a firmware update or reboot.

    ./push_effect.py aurora --host 192.168.2.149            # check, upload, verify
    ./push_effect.py aurora --host tree.local --select      # ...and switch to it
    ./push_effect.py _util rainbow_cycle --host tree.local  # several files, shared modules first
    ./push_effect.py aurora --dry-run                       # checks only, no network

Before uploading, each effect goes through the smoke checks (renders, power
budget, determinism, pure imports) and every file is compiled with mpy-cross.
If the device doesn't list the effect in its EFFECTS yet, effects/__init__.py
is pushed as well, last.

The device side is the effects OTA API described in docs/effect-api.md
("Pushing one effect over OTA"). tools/mock_tree.py implements it locally for
testing. --host defaults to $TREE_HOST; an auth token, if the firmware requires
one, comes from --token or $TREE_TOKEN.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import effects  # noqa: E402
import render  # noqa: E402
import smoke  # noqa: E402

SHARED_ORDER = ["_util", "_output", "_coords"]


class Device:
    def __init__(self, host, port, token, timeout):
        self.base = f"http://{host}:{port}"
        self.token = token
        self.timeout = timeout

    def request(self, method, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, _json(r.read())
        except urllib.error.HTTPError as e:
            return e.code, _json(e.read())

    def list(self):
        status, data = self.request("GET", "/effects")
        if status == 404:
            sys.exit(f"{self.base}/effects not found: this firmware has no effects OTA API yet "
                     "(see docs/effect-api.md); use ./flash.py --ota for a full update")
        if status != 200 or not isinstance(data.get("effects"), list):
            sys.exit(f"GET {self.base}/effects did not return the effects OTA listing "
                     f"(HTTP {status}); is this the tree?")
        return data

    def upload(self, module, source):
        digest = hashlib.sha256(source).hexdigest()
        status, data = self.request("POST", f"/effects/{module}", source, {
            "Content-Type": "text/x-python",
            "X-Content-SHA256": digest,
        })
        return status, data, digest


def _json(raw):
    try:
        return json.loads(raw.decode() or "{}")
    except ValueError:
        return {"raw": raw.decode(errors="replace")[:500]}


def local_checks(modules):
    rep = smoke.Report()
    coords, _ = render.find_coords()
    for m in modules:
        if m.startswith("_"):
            continue
        print(f"check {m}")
        mod = effects.load(m)
        smoke.check_metadata(rep, m, mod)
        smoke.check_imports(rep, m)
        smoke.check_render(rep, m, mod, coords, 3.0, 20)
        smoke.check_variants(rep, m, mod, coords, 20)
    smoke.check_mpy_cross(rep, required=False)
    return rep.failures


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modules", nargs="+", help="effect names or shared modules (_util, _output, _coords, __init__)")
    ap.add_argument("--host", default=os.environ.get("TREE_HOST"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", default=os.environ.get("TREE_TOKEN"))
    ap.add_argument("--select", action="store_true", help="switch the tree to the (last) pushed effect")
    ap.add_argument("--dry-run", action="store_true", help="run the local checks only")
    ap.add_argument("--no-check", action="store_true", help="skip the local checks (not recommended)")
    ap.add_argument("--timeout", type=float, default=15.0)
    args = ap.parse_args()

    modules = []
    for m in args.modules:
        m = Path(m).stem if m.endswith(".py") else m
        if not (ROOT / "effects" / f"{m}.py").exists():
            sys.exit(f"effects/{m}.py not found")
        if not m.startswith("_") and m not in effects.EFFECTS:
            sys.exit(f"{m!r} is not in EFFECTS (effects/__init__.py); register it first")
        if m not in modules:
            modules.append(m)

    if not args.no_check:
        failures = local_checks(modules)
        if failures:
            print("\nlocal checks failed; not pushing:")
            for f in failures:
                print(f"  {f}")
            return 1
        print("local checks passed")
    if args.dry_run:
        return 0
    if not args.host:
        sys.exit("--host (or $TREE_HOST) is required")

    dev = Device(args.host, args.port, args.token, args.timeout)
    before = dev.list()
    if any(not m.startswith("_") and m not in before.get("effects", []) for m in modules) \
            and "__init__" not in modules:
        modules.append("__init__")
    if "__init__" in modules:
        # Pushing the registry must not advertise effects the device lacks.
        have = set(before.get("files", {})) | set(modules)
        missing = [n for n in effects.EFFECTS if n not in have]
        if missing:
            sys.exit(f"effects/__init__.py lists {', '.join(missing)}, which the device doesn't have; "
                     f"push them too: ./push_effect.py {' '.join(modules[:-1] + missing)} ...")
    # Shared modules first so effects that need new helpers find them; the
    # registry last so the device never lists an effect it can't import.
    modules.sort(key=lambda m: (0, SHARED_ORDER.index(m)) if m in SHARED_ORDER
                 else (2, 0) if m == "__init__" else (1, 0))

    for m in modules:
        source = (ROOT / "effects" / f"{m}.py").read_bytes()
        status, data, digest = dev.upload(m, source)
        if status != 200 or not data.get("ok"):
            print(f"push {m}: FAILED (HTTP {status})\n  {data.get('error', data)}")
            print("  the device kept its previous version of this file")
            return 1
        if data.get("sha256") != digest:
            print(f"push {m}: device stored sha256 {data.get('sha256')}, expected {digest}")
            return 1
        print(f"push {m}: ok ({len(source)} bytes, sha256 {digest[:12]})")

    after = dev.list()
    files = after.get("files", {})
    for m in modules:
        want = hashlib.sha256((ROOT / "effects" / f"{m}.py").read_bytes()).hexdigest()
        if files.get(m) != want:
            print(f"verify {m}: device reports {files.get(m)}, expected {want}")
            return 1
    print(f"verified {len(modules)} file(s) on {dev.base}")

    if args.select:
        name = [m for m in modules if not m.startswith("_")][-1:]
        if not name:
            print("--select: no effect among the pushed files")
            return 1
        status, data = dev.request("POST", f"/effects/{name[0]}/select")
        if status != 200 or not data.get("ok"):
            print(f"select {name[0]}: FAILED (HTTP {status}) {data}")
            return 1
        print(f"selected {name[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
