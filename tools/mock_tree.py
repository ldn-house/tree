#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""A stand-in for the tree's effects OTA API, for testing push_effect.py and
as an executable reference for the firmware side (docs/effect-api.md).

    tools/mock_tree.py --port 8080 [--state DIR]

It keeps its own copy of effects/ in DIR (default .scratch/mock-tree), seeded
from the repository. Uploads are validated the way the device should validate
them: length and SHA-256, compile, import, construct and render one frame,
with the previous file restored on any failure.
"""

import argparse
import hashlib
import http.server
import json
import random
import re
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODULE = re.compile(r"^/effects/([A-Za-z_][A-Za-z0-9_]*)$")
SELECT = re.compile(r"^/effects/([a-z][a-z0-9_]*)/select$")


class Tree:
    def __init__(self, state):
        self.state = Path(state)
        pkg = self.state / "effects"
        if not pkg.exists():
            pkg.mkdir(parents=True)
            for f in (ROOT / "effects").glob("*.py"):
                shutil.copy(f, pkg / f.name)
        sys.path.insert(0, str(self.state))
        self.current = None

    def purge(self):
        for k in [k for k in sys.modules if k == "effects" or k.startswith("effects.")]:
            del sys.modules[k]

    def files(self):
        return {p.stem: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((self.state / "effects").glob("*.py"))}

    def listing(self):
        self.purge()
        import effects
        return {"effects": list(effects.EFFECTS), "files": self.files(), "current": self.current}

    def install(self, module, body):
        path = self.state / "effects" / f"{module}.py"
        backup = path.read_bytes() if path.exists() else None
        compile(body, f"effects/{module}.py", "exec")
        path.write_bytes(body)
        try:
            self.purge()
            import effects
            names = [module] if not module.startswith("_") else []
            if module == "__init__" or module.startswith("_"):
                names = [n for n in effects.EFFECTS if (self.state / "effects" / f"{n}.py").exists()]
            for name in names:
                mod = effects.load(name)
                ctx = effects.Ctx(500, None, effects.defaults(mod), random.Random(0))
                frame = bytearray(1500)
                mod.Effect(ctx).render(0, frame)
        except Exception:
            if backup is None:
                path.unlink()
            else:
                path.write_bytes(backup)
            self.purge()
            raise


def make_handler(tree):
    class Handler(http.server.BaseHTTPRequestHandler):
        def reply(self, status, data):
            body = json.dumps(data).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/effects":
                return self.reply(200, tree.listing())
            self.reply(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            m = SELECT.match(self.path)
            if m:
                if m.group(1) not in tree.listing()["effects"]:
                    return self.reply(404, {"ok": False, "error": f"unknown effect {m.group(1)}"})
                tree.current = m.group(1)
                return self.reply(200, {"ok": True, "current": tree.current})
            m = MODULE.match(self.path)
            if not m:
                return self.reply(404, {"ok": False, "error": "not found"})
            module = m.group(1)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length)
            if len(body) != length:
                return self.reply(400, {"ok": False, "error": "truncated body"})
            digest = hashlib.sha256(body).hexdigest()
            want = self.headers.get("X-Content-SHA256")
            if want and want != digest:
                return self.reply(400, {"ok": False, "error": f"sha256 mismatch: got {digest}"})
            try:
                tree.install(module, body)
            except Exception as e:  # noqa: BLE001
                return self.reply(422, {"ok": False, "error": "".join(
                    traceback.format_exception_only(type(e), e)).strip()})
            self.reply(200, {"ok": True, "file": f"effects/{module}.py", "bytes": length, "sha256": digest})

        def log_message(self, fmt, *args):
            print("mock-tree: " + fmt % args, flush=True)

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--state", default=str(ROOT / ".scratch" / "mock-tree"))
    args = ap.parse_args()
    tree = Tree(args.state)
    print(f"mock tree on http://127.0.0.1:{args.port} (state in {args.state})", flush=True)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(tree)).serve_forever()


if __name__ == "__main__":
    main()
