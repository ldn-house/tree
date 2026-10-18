#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "pillow"]
# ///
"""Hot-push effects to the tree over Wi-Fi, without a firmware update or reboot.

    ./push_effect.py aurora --host 192.168.2.149            # check, upload, verify
    ./push_effect.py aurora --host tree.local --activate    # ...and switch to it
    ./push_effect.py aurora sparkle --host tree.local       # several effects
    ./push_effect.py aurora --dry-run                       # checks only, no network
    ./push_effect.py aurora --host tree.local --delete      # drop the pushed copy

Before uploading, each effect goes through the smoke checks (renders, power
budget, determinism, pure imports) and is compiled with mpy-cross.

The device stores a pushed effect as an overlay that replaces (or adds to) the
firmware's copy until the next firmware install. Only effect modules can be
pushed; shared modules (effects/_util.py etc.) ship with firmware. A new
effect doesn't need registering on the device. See "Pushing one effect over
OTA" in docs/effect-api.md; tools/mock_tree.py is a local stand-in.

--host defaults to $TREE_HOST. Like flash.py, the HTTP token comes from
--token, $TREE_TOKEN or HTTP_TOKEN in config.py, and when config.py sets
UPDATE_KEY each upload is signed: X-Signature is the hex HMAC-SHA256, keyed
with UPDATE_KEY, of b"tree-effect\n" + name + b"\n" + body. A device with
UPDATE_KEY set refuses unsigned or wrongly signed effects (403) before
running any of their code.
"""

import argparse
import hashlib
import hmac
import json
import os
import runpy
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import effects  # noqa: E402
import render  # noqa: E402
import smoke  # noqa: E402


def load_config(path):
    """Upper-case settings from config.py, as flash.py reads them."""
    path = Path(path)
    if not path.exists():
        return {}
    return {k: v for k, v in runpy.run_path(str(path)).items() if k.isupper()}


def sign(key, name, body):
    """X-Signature for an effect push (firmware bootmgr.signature_ok("effect", ...))."""
    msg = b"tree-effect\n" + name.encode() + b"\n" + body
    return hmac.new(key.encode(), msg, hashlib.sha256).hexdigest()


class Device:
    def __init__(self, host, port, token, timeout, update_key=None):
        self.base = f"http://{host}:{port}"
        self.token = token
        self.timeout = timeout
        self.update_key = update_key

    def request(self, method, path, body=None, headers=None):
        req = urllib.request.Request(self.base + path, data=body, method=method, headers=headers or {})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, _json(r.read())
        except urllib.error.HTTPError as e:
            return e.code, _json(e.read())

    def effects(self):
        """{name: entry} from GET /effects."""
        status, data = self.request("GET", "/effects")
        if status == 404:
            sys.exit(f"{self.base}/effects not found: this firmware has no effects API "
                     "(see docs/effect-api.md); use ./flash.py --ota for a full update")
        listing = data.get("effects") if isinstance(data, dict) else None
        if status != 200 or not isinstance(listing, list):
            sys.exit(f"GET {self.base}/effects did not return the effects listing (HTTP {status}); is this the tree?")
        return {e.get("name"): e for e in listing if isinstance(e, dict)}

    def push(self, name, body, mpy, activate):
        query = []
        if mpy:
            query.append("format=mpy")
        if activate:
            query.append("activate=1")
        path = f"/effects/{name}" + ("?" + "&".join(query) if query else "")
        headers = {
            "Content-Type": "application/octet-stream" if mpy else "text/x-python",
            "X-Content-SHA256": hashlib.sha256(body).hexdigest(),
        }
        if self.update_key:
            headers["X-Signature"] = sign(self.update_key, name, body)
        return self.request("PUT", path, body, headers)


def _json(raw):
    try:
        return json.loads(raw.decode() or "{}")
    except ValueError:
        return {"error": raw.decode(errors="replace")[:500]}


def local_checks(names):
    rep = smoke.Report()
    coords, _ = render.find_coords()
    for name in names:
        print(f"check {name}")
        mod = effects.load(name)
        smoke.check_metadata(rep, name, mod)
        smoke.check_imports(rep, name)
        smoke.check_render(rep, name, mod, coords, 3.0, 20)
        smoke.check_variants(rep, name, mod, coords, 20)
    smoke.check_mpy_cross(rep, required=False)
    return rep.failures


def compile_mpy(name):
    uvx = shutil.which("uvx")
    if not uvx:
        sys.exit("--mpy needs uvx (for mpy-cross)")
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / f"{name}.mpy"
        subprocess.run([uvx, "--from", "mpy-cross", "mpy-cross", "-o", str(out), f"effects/{name}.py"],
                       cwd=ROOT, check=True)
        return out.read_bytes()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="effect names (or effects/<name>.py paths)")
    ap.add_argument("--host", default=os.environ.get("TREE_HOST"))
    ap.add_argument("--port", type=int, help="HTTP port (default: HTTP_PORT/OTA_PORT in config.py, else 8080)")
    ap.add_argument("--token", default=os.environ.get("TREE_TOKEN"), help="HTTP token ($TREE_TOKEN, else config.py)")
    ap.add_argument("--config", default=str(ROOT / "config.py"), help="config.py to read HTTP_TOKEN and UPDATE_KEY from")
    ap.add_argument("--activate", "--select", action="store_true", help="switch the tree to the last effect pushed")
    ap.add_argument("--mpy", action="store_true",
                    help="push bytecode compiled by mpy-cross (must match the device's MicroPython .mpy version)")
    ap.add_argument("--delete", action="store_true", help="remove the pushed overlay, reverting to the firmware copy")
    ap.add_argument("--dry-run", action="store_true", help="run the local checks only")
    ap.add_argument("--no-check", action="store_true", help="skip the local checks (not recommended)")
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()

    names = []
    for n in args.names:
        n = Path(n).stem if n.endswith(".py") else n
        if n.startswith("_"):
            sys.exit(f"{n} is a shared module; shared modules ship with firmware (./flash.py), not hot-push")
        if not args.delete and not (ROOT / "effects" / f"{n}.py").exists():
            sys.exit(f"effects/{n}.py not found")
        if n not in names:
            names.append(n)

    if not args.delete and not args.no_check:
        failures = local_checks(names)
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

    cfg = load_config(args.config)
    token = args.token or cfg.get("HTTP_TOKEN")
    key = cfg.get("UPDATE_KEY")
    port = args.port or cfg.get("HTTP_PORT") or cfg.get("OTA_PORT") or 8080
    if not key and not args.delete:
        print("warning: no UPDATE_KEY in config.py; pushing unsigned (a device with UPDATE_KEY set refuses this)")
    dev = Device(args.host, port, token, args.timeout, key)
    dev.effects()  # fail early if this isn't the tree

    if args.delete:
        for n in names:
            status, data = dev.request("DELETE", f"/effects/{n}")
            print(f"delete {n}: " + ("ok" if status == 200 else f"FAILED (HTTP {status}) {data.get('error', data)}"))
            if status != 200:
                return 1
        return 0

    for k, n in enumerate(names):
        body = compile_mpy(n) if args.mpy else (ROOT / "effects" / f"{n}.py").read_bytes()
        activate = args.activate and k == len(names) - 1
        status, data = dev.push(n, body, args.mpy, activate)
        if status == 401:
            print(f"push {n}: unauthorised (HTTP 401); set HTTP_TOKEN in config.py, $TREE_TOKEN or --token")
            return 1
        if status == 403:
            print(f"push {n}: refused (HTTP 403): {data.get('error', data)}\n"
                  "  the device requires a signature; check UPDATE_KEY in config.py matches the device's")
            return 1
        if status != 200 or data.get("effect") != n:
            print(f"push {n}: FAILED (HTTP {status})\n  {data.get('error', data)}")
            print("  the device kept its previous version of this effect")
            return 1
        print(f"push {n}: ok ({len(body)} bytes{', activated' if data.get('activated') else ''})")

    listing = dev.effects()
    for n in names:
        entry = listing.get(n)
        if not entry or not entry.get("overlay") or entry.get("error"):
            print(f"verify {n}: device reports {entry}")
            return 1
    print(f"verified {len(names)} effect(s) on {dev.base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
