"""Board-sized heap campaign (review: "board heap and fragmentation untested").

Runs the firmware on a 32-bit MicroPython unix build (objects the size they
are on the RP2350) with a heap the size the Pico 2 W actually has free
(452,544 bytes idle at the REPL, measured on the user's board, minus margin
for Wi-Fi), the real effects, the full coordinates file and TLS:

1. cycle through every real effect (auto-cycle every few seconds), three
   times over, with MQTT and Home Assistant connected to a local mosquitto
   (--mosquitto): the free heap must never drop below --floor (64 KB), even
   uncollected, and the MQTT session must never drop;
2. meanwhile check for updates and then self-update over HTTPS (mbedtls on
   the MicroPython heap, as on the board) from a local TLS server;
3. after the reboot, cycle again and play the default calibration sequence.

Stdlib only, so it runs inside a 32-bit container:
    python3 heapcheck.py --micropython micropython32 --bundle a.bin --update b.bin \
        --cert c.pem --key k.pem --heap 400K --out result.json
Exit status 0 when every step succeeded without MemoryError.
"""

import argparse
import http.server
import json
import os
import ssl
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--micropython", required=True)
ap.add_argument("--bundle", required=True)
ap.add_argument("--update", required=True, help="newer signed bundle served over HTTPS")
ap.add_argument("--key-string", default="heap-signing-key")
ap.add_argument("--cert", required=True)
ap.add_argument("--key", required=True)
ap.add_argument("--heap", default="400K")
ap.add_argument("--root", default="heapcheck-device")
ap.add_argument("--calib", help="a .tcal to install as the default sequence")
ap.add_argument("--out", default="heapcheck.json")
ap.add_argument("--mosquitto", help="mosquitto binary: run with MQTT + Home Assistant connected")
ap.add_argument("--floor", type=int, default=64 * 1024, help="lowest acceptable free heap, even uncollected")
ap.add_argument("--loops", type=int, default=3, help="auto-cycle loops per boot")
args = ap.parse_args()

os.environ["MICROPYTHON"] = args.micropython
os.environ["DEVSIM_HEAP"] = args.heap
sys.path.insert(0, str(Path(__file__).resolve().parent))
import devsim  # noqa: E402

update = Path(args.update)
files = {"/main/tree-firmware.bin": update.read_bytes(), "/main/manifest.json": (update.parent / "manifest.json").read_bytes()}


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        data = files.get(self.path)
        if data is None:
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
ctx.load_cert_chain(args.cert, args.key)
srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
threading.Thread(target=srv.serve_forever, daemon=True).start()
url = "https://127.0.0.1:%d/{channel}/{asset}" % srv.server_address[1]

result = {"heap": args.heap, "floor": args.floor, "steps": {}, "effects_seen": [], "errors": {},
          "heap_low_uncollected": None, "heap_live_min": None, "checkpoints": {}}
broker = None
mqtt = {}
if args.mosquitto:
    import subprocess

    port = devsim.free_port()
    Path(args.root).mkdir(parents=True, exist_ok=True)
    conf = Path(args.root) / "mosquitto.conf"
    conf.write_text("listener %d 127.0.0.1\nallow_anonymous true\npersistence false\n" % port)
    broker = subprocess.Popen([args.mosquitto, "-c", str(conf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1)
    mqtt = {"MQTT_BROKER": "127.0.0.1", "MQTT_PORT": port, "MQTT_KEEPALIVE": 10}
d = devsim.Device(Path(args.root) / "dev", UPDATE_KEY=args.key_string, UPDATE_URL=url, TRIAL_HEALTHY_S=5,
                  TRIAL_DEADLINE_S=120, **mqtt)
d.install_recovery()
d.usb_install(Path(args.bundle))
d.start()


def st():
    return d.get("/", timeout=10)[1]


def track(seconds):
    seen = set()
    end = time.time() + seconds
    while time.time() < end:
        try:
            s = st()
            seen.add(s["effect"]["running"])
            result["errors"].update(s["effect"]["failed"])
            for k in ("heap_low_uncollected", "heap_live_min"):
                m = s["supervisor"][k]
                if m is not None and (result[k] is None or m < result[k]):
                    result[k] = m
        except Exception:
            pass
        time.sleep(0.5)
    return seen


def checkpoint(name):
    """The boot's low-water marks so far (they include start-up and every MQTT
    publish), against the floor; and no MQTT session may have dropped."""
    s = st()
    sup = s["supervisor"]
    cp = {"heap_low_uncollected": sup["heap_low_uncollected"], "heap_live_min": sup["heap_live_min"],
          "heap_sites": sup["heap_sites"], "stall_max_ms": sup["stall_max_ms"], "mqtt": s["mqtt"]}
    result["checkpoints"][name] = cp
    good = sup["heap_low_uncollected"] >= args.floor
    if s["mqtt"] is not None:
        good = good and s["mqtt"]["connected"] and s["mqtt"]["connects"] == 1
    result["steps"][name] = good


ok = True
try:
    s = d.wait_http(60, pred=lambda s: not s["boot"]["trial"])
    names = s["effects"]
    result["effects"] = names
    d.post("/state", {"auto_s": 4, "brightness": 255})
    seen = track(args.loops * 4 * len(names) + 8)
    result["effects_seen"] = sorted(x for x in seen if x)
    result["steps"]["cycle_all_effects"] = set(names) <= seen
    checkpoint("floor_and_mqtt_after_cycles")
    code, chk = d.request("GET", "/update/github/check", timeout=60)
    result["steps"]["tls_check"] = code == 200 and chk.get("latest") is not None and not chk.get("error")
    result["check"] = chk
    code, _ = d.post("/update/github?channel=main")
    track(3)
    s = d.wait_http(180, pred=lambda s: s["slot"] == "b" and not s["boot"]["trial"])
    result["steps"]["tls_self_update"] = s["version"] != "" and s["slot"] == "b"
    seen = track(args.loops * 4 * len(names) + 8)
    result["steps"]["cycle_after_update"] = set(names) <= seen
    checkpoint("floor_and_mqtt_after_update")
    if args.calib:
        (d.root / "fw" / "b" / "data" / "calib_default.tcal").write_bytes(Path(args.calib).read_bytes())
        code, res = d.post("/calibrate/play?sequence=default&loops=1")
        track(res["loop_ms"] / 1000 + 3 if code == 200 else 1)
        log = d.get("/calibrate/log")[1]
        result["steps"]["calibration_default"] = code == 200 and log["frames_shown"] > 0 and log["frames_skipped"] == 0
        result["calib_log"] = log["late_us"]
finally:
    d.stop()
    srv.shutdown()
    if broker:
        broker.terminate()

console = d.console()
result["memory_errors"] = console.count("MemoryError")
result["boots"] = d.boots
ok = all(result["steps"].values()) and not result["errors"] and result["memory_errors"] == 0
result["ok"] = ok
Path(args.out).write_text(json.dumps(result, indent=1))
print(json.dumps(result, indent=1))
sys.exit(0 if ok else 1)
