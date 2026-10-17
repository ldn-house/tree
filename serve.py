#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Serve the effects viewer (viewer/) with hot reload.

    ./serve.py               # http://localhost:8000/viewer/
    ./serve.py --port 9000 --no-browser

Serves the repository root without caching, plus /api/mtimes (modification
times of effects/*.py) which the viewer polls to reload changed effects.
"""

import argparse
import http.server
import json
import os
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/api/mtimes":
            body = json.dumps(
                {p.name: p.stat().st_mtime for p in sorted((ROOT / "effects").glob("*.py"))}
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/viewer/")
            self.end_headers()
            return
        super().do_GET()

    def log_message(self, fmt, *args):
        if os.environ.get("SERVE_VERBOSE"):
            super().log_message(fmt, *args)


class Server(http.server.ThreadingHTTPServer):
    # The viewer fetches the whole effects/ package at once; the default
    # backlog of 5 drops some of those connections.
    request_queue_size = 64
    daemon_threads = True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    with Server(("127.0.0.1", args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}/viewer/"
        print(f"Serving {url}")
        if not args.no_browser:
            webbrowser.open(url)
        httpd.serve_forever()


if __name__ == "__main__":
    main()
