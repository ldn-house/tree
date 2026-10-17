#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpremote", "httpx", "mpy-cross==1.29.0.post2"]
# ///
"""Install firmware on the tree's Pico, over USB or Wi-Fi (OTA).

USB (also installs the recovery layer, which OTA can never change):
    ./flash.py                       build locally, install recovery layer + firmware
    ./flash.py --config              ...and copy config.py
    ./flash.py --coords              ...and copy coordinates/coords_compact.txt
    ./flash.py --github              install the latest GitHub Release instead of a local build

OTA (needs HTTP_TOKEN, from config.py, $TREE_TOKEN or --token):
    ./flash.py --ota --host tree.local                 build locally and push
    ./flash.py --ota --host tree.local --github        push the latest release (--channel main|stable)
    ./flash.py --ota --host tree.local --coords        push coordinates only
    ./flash.py --ota --host tree.local --status        show status (read-only)
    ./flash.py --ota --host tree.local --rollback      boot the previous firmware

An OTA update is installed into the inactive slot, verified on the device and
booted as a trial; flash.py waits until the device confirms it healthy, or
reports the automatic rollback.
"""

import argparse
import hashlib
import hmac
import json
import os
import runpy
import secrets
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RECOVERY = ["bootmgr.py", "safemode.py", "boot.py", "main.py"]
REPO = "ldn-house/tree"


def load_config():
    path = ROOT / "config.py"
    if not path.exists():
        return {}
    return {k: v for k, v in runpy.run_path(str(path)).items() if k.isupper()}


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)


# -- bundles -----------------------------------------------------------------


def build_local(out: Path, key):
    cmd = [sys.executable, str(ROOT / "tools" / "build_firmware.py"), "--out", str(out)]
    env = dict(os.environ)
    if key:
        env["FIRMWARE_SIGNING_KEY"] = key
    subprocess.run(cmd, check=True, env=env)
    return out / "tree-firmware.bin"


def release_url(channel, asset):
    if channel == "stable":
        return f"https://github.com/{REPO}/releases/latest/download/{asset}"
    return f"https://github.com/{REPO}/releases/download/firmware-{channel}/{asset}"


def fetch_release(out: Path, channel):
    import httpx

    url = release_url(channel, "tree-firmware.bin")
    print(f"Downloading {url}")
    try:
        r = httpx.get(url, follow_redirects=True, timeout=60)
        r.raise_for_status()
    except httpx.HTTPError as e:
        die(f"could not download the {channel} release: {e}\nSee https://github.com/{REPO}/releases")
    path = out / "tree-firmware.bin"
    path.write_bytes(r.content)
    return path


def inspect(bundle: Path, key=None):
    """Parse and check a bundle on the host. Returns the manifest."""
    data = bundle.read_bytes()
    if data[:8] != b"TREEFW1\n":
        die(f"{bundle} is not a firmware bundle")
    mlen = struct.unpack(">I", data[8:12])[0]
    mb = data[12 : 12 + mlen]
    m = json.loads(mb)
    slen = struct.unpack(">I", data[12 + mlen : 16 + mlen])[0]
    sig = data[16 + mlen : 16 + mlen + slen]
    pos = 16 + mlen + slen
    for f in m["files"]:
        chunk = data[pos : pos + f["size"]]
        if hashlib.sha256(chunk).hexdigest() != f["sha256"]:
            die(f"bundle is corrupt: {f['path']} does not match its hash")
        pos += f["size"]
    if pos != len(data):
        die("bundle has trailing data")
    if key and sig and not hmac.compare_digest(sig, hmac.new(key.encode(), mb, hashlib.sha256).digest()):
        die("bundle signature does not match UPDATE_KEY in config.py")
    if key and not sig:
        print("warning: bundle is unsigned; a device with UPDATE_KEY set will refuse it over OTA")
    m["signed"] = bool(sig)
    return m


# -- USB -----------------------------------------------------------------------

# Run first in the same mpremote session: whatever firmware was running had
# the watchdog going, and mpremote's soft reset stopped whatever fed it.
FEEDER = "import machine\n_w = machine.WDT(timeout=8000)\n_t = machine.Timer(period=1000, callback=lambda t: _w.feed())"


def usb(bundle: Path, extra_files):
    cmd = ["mpremote", "exec", FEEDER]
    for f in RECOVERY:
        cmd += ["+", "cp", str(ROOT / "firmware" / f), f":{f}"]
    for src, dest in extra_files:
        cmd += ["+", "cp", str(src), f":{dest}"]
    cmd += ["+", "cp", str(bundle), ":_usb.bin"]
    cmd += ["+", "exec", "import bootmgr\nbootmgr.install_file('_usb.bin')\nimport os\nos.remove('_usb.bin')"]
    cmd += ["+", "reset"]
    print("Installing over USB (recovery layer, then firmware)...")
    subprocess.run(cmd, check=True)
    print("Done. The board is rebooting into the new firmware.")


# -- OTA -----------------------------------------------------------------------


class Tree:
    def __init__(self, host, port, token):
        import httpx

        self.base = f"http://{host}:{port}"
        self.h = {"Authorization": f"Bearer {token}"} if token else {}
        self.httpx = httpx

    def status(self, timeout=5.0):
        r = self.httpx.get(self.base + "/", timeout=timeout)
        r.raise_for_status()
        return r.json()

    def call(self, method, path, **kw):
        headers = {**self.h, **kw.pop("headers", {})}
        r = self.httpx.request(method, self.base + path, headers=headers, timeout=kw.pop("timeout", 30.0), **kw)
        try:
            body = r.json()
        except ValueError:
            body = {"error": r.text}
        if r.status_code >= 400:
            die(f"{method} {path}: HTTP {r.status_code}: {body.get('error', body)}")
        return body

    def wait(self, pred, timeout):
        end = time.time() + timeout
        while time.time() < end:
            try:
                st = self.status(3)
                if pred(st):
                    return st
            except (self.httpx.HTTPError, OSError, ValueError):
                pass
            time.sleep(2)
        return None


def ota_update(tree: Tree, bundle: Path, manifest):
    try:
        before = tree.status()
    except Exception as e:
        die(f"cannot reach {tree.base}: {e}\nCheck power, Wi-Fi and the address.")
    if before.get("mode") == "safe":
        print(f"Device is in safe mode ({before['reason']}); installing anyway.")
    else:
        print(f"Device runs {before['version']} (slot {before['slot']}); installing {manifest['version']}.")
    data = bundle.read_bytes()
    res = tree.call("POST", "/update", content=data, headers={"Content-Type": "application/octet-stream"}, timeout=120.0)
    print(f"Installed {res.get('installed')}; rebooting into it as a trial.")
    st = tree.wait(lambda s: s.get("version") == manifest["version"] or "rolled back" in s.get("boot", {}).get("status", ""),
                   120)
    if st is None:
        die("the device did not come back within 2 minutes")
    if st.get("version") != manifest["version"]:
        die(f"the device rolled back: {st['boot']['status']}")
    print("Booted; waiting for it to confirm the update healthy...")
    st = tree.wait(lambda s: not s.get("boot", {}).get("trial") or s.get("version") != manifest["version"], 420)
    if st is None:
        die("no confirmation within 7 minutes; check GET / on the device")
    if st.get("version") != manifest["version"]:
        die(f"the update failed its health check and was rolled back: {st['boot']['status']}")
    print(f"OK: {manifest['version']} confirmed healthy.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ota", action="store_true", help="use Wi-Fi instead of USB")
    ap.add_argument("--host", default=os.environ.get("TREE_HOST"), help="device address for --ota ($TREE_HOST)")
    ap.add_argument("--port", type=int, default=None, help="HTTP port (default: HTTP_PORT from config.py or 8080)")
    ap.add_argument("--token", default=os.environ.get("TREE_TOKEN"), help="HTTP token ($TREE_TOKEN, else config.py)")
    ap.add_argument("--github", action="store_true", help="use the latest GitHub Release, not a local build")
    ap.add_argument("--channel", default="main", choices=["main", "stable"], help="release channel for --github")
    ap.add_argument("--bundle", type=Path, help="use this bundle file")
    ap.add_argument("--config", action="store_true", help="USB: also copy config.py")
    ap.add_argument("--all", action="store_true", help="USB: same as --config --coords")
    ap.add_argument("--coords", action="store_true", help="also install coordinates/coords_compact.txt")
    ap.add_argument("--status", action="store_true", help="OTA: print device status and exit")
    ap.add_argument("--rollback", action="store_true", help="OTA: boot the previous firmware")
    ap.add_argument("--reboot", action="store_true", help="OTA: reboot the device")
    args = ap.parse_args()

    cfg = load_config()
    token = args.token or cfg.get("HTTP_TOKEN")
    key = cfg.get("UPDATE_KEY")
    port = args.port or cfg.get("HTTP_PORT") or cfg.get("OTA_PORT") or 8080
    if args.all:
        args.config = args.coords = True
    coords = ROOT / "coordinates" / "coords_compact.txt"

    if args.ota:
        if not args.host:
            ap.error("--host (or $TREE_HOST) is required with --ota")
        tree = Tree(args.host, port, token)
        if args.status:
            print(json.dumps(tree.status(), indent=2))
            return
        if not token:
            die("no token: set HTTP_TOKEN in config.py, $TREE_TOKEN or --token")
        if args.rollback:
            print(tree.call("POST", "/update/rollback"))
            return
        if args.reboot:
            print(tree.call("POST", "/reboot"))
            return
        if args.config:
            die("config.py can only be changed over USB (it holds the token and Wi-Fi settings)")
        if args.coords:
            res = tree.call("PUT", "/files/coords_compact.txt", content=coords.read_bytes())
            print(f"Coordinates stored: {res['known']} of {res['leds']} LEDs placed.")
            if not (args.github or args.bundle):
                return

    if not args.ota:
        if not shutil.which("mpremote"):
            die("mpremote not found")
        if not cfg:
            die("config.py is missing: copy config.example.py and fill it in")
        if not cfg.get("HTTP_TOKEN"):
            die("config.py has no HTTP_TOKEN, so the device would refuse every OTA request.\n"
                f"Add a random one, for example:\n    HTTP_TOKEN = \"{secrets.token_urlsafe(24)}\"")

    with tempfile.TemporaryDirectory(prefix="tree-fw-") as tmp:
        tmp = Path(tmp)
        if args.bundle:
            bundle = args.bundle
        elif args.github:
            bundle = fetch_release(tmp, args.channel)
        else:
            bundle = build_local(tmp, key)
        manifest = inspect(bundle, key)
        print(f"Bundle {manifest['version']} ({manifest.get('ref')}, built {manifest.get('built')}), "
              f"{len(manifest['files'])} files, {'signed' if manifest['signed'] else 'unsigned'}")
        if args.ota:
            ota_update(Tree(args.host, port, token), bundle, manifest)
        else:
            extra = []
            if args.config:
                extra.append((ROOT / "config.py", "config.py"))
            if args.coords:
                extra.append((coords, "coords_compact.txt"))
            usb(bundle, extra)


if __name__ == "__main__":
    main()
