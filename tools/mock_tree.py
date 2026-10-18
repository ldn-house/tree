#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""A local stand-in for the tree's effects API, for testing push_effect.py
without hardware. It mirrors the firmware's endpoints (firmware/app/api.py):

    GET    /effects                list effects (bundled + overlays)
    PUT    /effects/<name>         hot-push an effect (also POST); ?activate=1
    DELETE /effects/<name>         remove a pushed overlay

    tools/mock_tree.py --port 8080 [--token SECRET] [--update-key KEY]

Bundled effects come from this checkout's effects/; pushed ones are kept in
memory. An upload is validated like the device does it: optional
X-Content-SHA256, compile, import under a staging name, build Effect(ctx) and
render t=0 and t=33; only then does it replace the overlay. With
--update-key, a push must carry X-Signature = hex HMAC-SHA256(KEY,
b"tree-effect\n" + name + b"\n" + body), checked before any code runs.
"""

import argparse
import hashlib
import hmac
import http.server
import json
import random
import re
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import effects  # noqa: E402

NAME = re.compile(r"^/effects/([a-z][a-z0-9_]*)(\?.*)?$")
MAX_EFFECT = 32 * 1024
N = 500


class Tree:
    def __init__(self):
        self.overlays = {}  # name -> module
        self.running = effects.EFFECTS[0]

    def module(self, name):
        return self.overlays.get(name) or effects.load(name)

    def names(self):
        return list(effects.EFFECTS) + [n for n in self.overlays if n not in effects.EFFECTS]

    def listing(self):
        out = []
        for name in self.names():
            mod = self.module(name)
            out.append({
                "name": name,
                "duration_ms": mod.DURATION_MS,
                "needs_coords": mod.NEEDS_COORDS,
                "params": {k: {"default": v[0], "min": v[1], "max": v[2], "value": v[0]}
                           for k, v in getattr(mod, "PARAMS", {}).items()},
                "overlay": name in self.overlays,
                "eligible": True,
                "error": None,
            })
        return {"effects": out, "running": self.running, "fx_error": None}

    def push(self, name, source):
        code = compile(source, f"fx_{name}.py", "exec")
        mod = types.ModuleType(f"fx_{name}__new")
        exec(code, mod.__dict__)
        if not hasattr(mod, "Effect"):
            raise ValueError("module has no Effect class")
        params = {k: v[0] for k, v in getattr(mod, "PARAMS", {}).items()}
        inst = mod.Effect(effects.Ctx(N, None, params, random.Random(0)))
        frame = bytearray(3 * N)
        inst.render(0, frame)
        inst.render(33, frame)
        self.overlays[name] = mod
        return {"duration_ms": mod.DURATION_MS, "params": getattr(mod, "PARAMS", {}),
                "needs_coords": mod.NEEDS_COORDS}


def signature_ok(key, kind, body, sig_hex):
    """Mirror of firmware bootmgr.signature_ok: True if no key is set."""
    if not key:
        return True
    if not sig_hex:
        return False
    try:
        sig = bytes.fromhex(sig_hex.strip())
    except ValueError:
        return False
    want = hmac.new(key.encode(), b"tree-" + kind.encode() + b"\n" + body, hashlib.sha256).digest()
    return hmac.compare_digest(sig, want)


def make_handler(tree, token, update_key=None):
    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def authorised(self):
            if not token:
                return True
            got = self.headers.get("Authorization", "").removeprefix("Bearer ").strip() or \
                self.headers.get("X-Tree-Token", "")
            if got != token:
                self.reply(401, {"error": "unauthorised"})
                return False
            return True

        def do_GET(self):
            if self.path == "/effects":
                return self.reply(200, tree.listing())
            self.reply(404, {"error": "not found"})

        def do_PUT(self):
            m = NAME.match(self.path)
            if not m:
                return self.reply(400, {"error": "effect names are lowercase letters, digits and _"})
            if not self.authorised():
                return
            name = m.group(1)
            query = dict(kv.split("=", 1) for kv in (m.group(2) or "?")[1:].split("&") if "=" in kv)
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_EFFECT:
                return self.reply(413, {"error": "too large"})
            data = self.rfile.read(length)
            want = self.headers.get("X-Content-SHA256")
            if want and hashlib.sha256(data).hexdigest() != want.strip().lower():
                return self.reply(400, {"error": "X-Content-SHA256 does not match the body"})
            if not signature_ok(update_key, "effect", name.encode() + b"\n" + data, self.headers.get("X-Signature")):
                return self.reply(403, {"error": "effects must be signed with UPDATE_KEY (X-Signature header)"})
            if query.get("format") == "mpy" or (len(data) > 4 and data[0] == 0x4D and data[1] < 32):
                return self.reply(400, {"error": "mock tree cannot run .mpy; push source"})
            try:
                result = tree.push(name, data.decode())
            except SyntaxError as e:
                return self.reply(400, {"error": f"{e!r}"})
            except Exception as e:  # noqa: BLE001
                return self.reply(400, {"error": f"effect failed to load or render: {e!r}"})
            activate = query.get("activate") in ("1", "true", "yes")
            if activate:
                tree.running = name
            result.update({"effect": name, "overlay": True, "activated": activate})
            self.reply(200, result)

        do_POST = do_PUT

        def do_DELETE(self):
            m = NAME.match(self.path)
            if not m:
                return self.reply(404, {"error": "not found"})
            if not self.authorised():
                return
            if tree.overlays.pop(m.group(1), None) is None:
                return self.reply(404, {"error": "no hot-pushed effect named " + m.group(1)})
            self.reply(200, {"effect": m.group(1), "overlay": False})

        def log_message(self, fmt, *args):
            print("mock-tree: " + fmt % args, flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--token", help="require this bearer token on writes")
    ap.add_argument("--update-key", help="require effect pushes signed with this UPDATE_KEY")
    args = ap.parse_args()
    print(f"mock tree on http://127.0.0.1:{args.port}", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(Tree(), args.token, args.update_key)).serve_forever()


if __name__ == "__main__":
    main()
