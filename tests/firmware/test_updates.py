"""OTA updates over HTTP, exercised with curl against the simulated device."""

import json
import struct
import time

import pytest

from conftest import SIGNING_KEY, curl
from devsim import wait_for


def push(d, bundle, *extra, token=True, path="/update"):
    args = ["-X", "POST", "--data-binary", f"@{bundle}", "-H", "Content-Type: application/octet-stream"]
    if token:
        args += ["-H", f"Authorization: Bearer {d.token if token is True else token}"]
    return curl(*args, *extra, d.url(path))


def running(d, version, timeout=60, confirmed=None):
    def ok(st):
        if st.get("version") != version:
            return False
        if confirmed is not None and st["boot"]["trial"] == confirmed:
            return False
        return True

    return d.wait_http(timeout, pred=ok)


def test_auth_is_required(device, bundles):
    d = device
    code, body = push(d, bundles["v2"], token=False)
    assert code == 401 and "token" in body["error"]
    code, body = push(d, bundles["v2"], token="wrong")
    assert code == 401
    for method, path in (("POST", "/reboot"), ("POST", "/state"), ("POST", "/update/rollback"),
                         ("PUT", "/effects/x"), ("GET", "/calibrate/start"), ("POST", "/calibrate/play")):
        code, _ = curl("-X", method, d.url(path))
        assert code == 401, (method, path, code)
    # Read-only endpoints stay open.
    assert curl(d.url("/"))[0] == 200
    assert curl(d.url("/calibrate"))[0] == 200
    assert d.get("/")[1]["version"] == "v1"


def test_no_token_configured_fails_closed(make_device, bundles):
    d = make_device(bundles["v1"], HTTP_TOKEN=None)
    d.wait_http(30)
    code, body = push(d, bundles["v2"])
    assert code == 403 and "HTTP_TOKEN" in body["error"]


def test_truncated_upload_is_not_installed(device, bundles, tmp_path):
    d = device
    before = d.boot_state()
    data = bundles["v2"].read_bytes()
    half = tmp_path / "half.bin"
    half.write_bytes(data[: len(data) // 2])
    # Claim the full length, send half, then give up (client vanishes).
    code, _ = curl("-X", "POST", "--data-binary", f"@{half}", "-H", f"Content-Length: {len(data)}",
                   "-H", f"Authorization: Bearer {d.token}", "--max-time", "3", d.url("/update"))
    assert code == 0  # curl timed out; no response
    st = wait_for(lambda: (lambda s: s if s["update"]["state"] == "error" else None)(d.get("/")[1]), 20,
                  what="update error")
    assert "truncated" in st["update"]["error"] or "timeout" in st["update"]["error"].lower() or st["update"]["error"]
    assert d.boot_state()["active"] == before["active"]
    assert not (d.root / "fw" / "b" / "manifest.json").exists()
    time.sleep(2)
    assert d.get("/")[1]["version"] == "v1" and len(d.boots) == 0

    # Body shorter than Content-Length with the connection closed promptly.
    code, body = curl("-X", "POST", "--data-binary", f"@{half}", "-H", f"Authorization: Bearer {d.token}",
                      d.url("/update"))
    assert code == 400 and "Content-Length does not match" in body["error"]


def test_bad_hash_is_rejected(device, bundles, tmp_path):
    d = device
    data = bytearray(bundles["v2"].read_bytes())
    data[-100] ^= 0xFF  # flip a byte inside the last file
    bad = tmp_path / "bad.bin"
    bad.write_bytes(bytes(data))
    code, body = push(d, bad)
    assert code == 400 and "sha256 mismatch" in body["error"], body
    assert d.boot_state()["active"] == "a" and not (d.root / "fw" / "b" / "manifest.json").exists()

    # A manifest that lies about a size is caught before anything is written.
    raw = bundles["v2"].read_bytes()
    mlen = struct.unpack(">I", raw[8:12])[0]
    m = json.loads(raw[12 : 12 + mlen])
    m["files"][0]["size"] += 1
    mb = json.dumps(m).encode()
    lie = tmp_path / "lie.bin"
    lie.write_bytes(raw[:8] + struct.pack(">I", len(mb)) + mb + raw[12 + mlen :])
    code, body = push(d, lie)
    assert code == 400 and "Content-Length" in body["error"]

    code, body = push(d, bundles["v2"], path="/update?dry_run=1")
    assert code == 200 and body["verified"] == "v2"
    assert d.boot_state()["active"] == "a" and not (d.root / "fw" / "b" / "manifest.json").exists()


def test_valid_update_is_confirmed_and_keeps_previous(device, bundles):
    d = device
    running(d, "v1", confirmed=True)
    code, body = push(d, bundles["v2"])
    assert code == 200 and body["installed"] == "v2" and body["rebooting"], body
    st = running(d, "v2")
    assert st["boot"]["trial"] and st["slot"] == "b"
    # A second update is refused until this one is confirmed.
    code, body = push(d, bundles["v1"])
    assert code == 409
    st = running(d, "v2", confirmed=True)
    bs = d.boot_state()
    assert bs["active"] == "b" and bs["previous"] == "a" and not bs["trial"] and bs["boots"] == 0
    assert "confirmed b(v2)" in bs["events"]
    # Manual rollback to the kept slot.
    code, body = curl("-X", "POST", "-H", f"Authorization: Bearer {d.token}", d.url("/update/rollback"))
    assert code == 200 and body["slot"] == "a"
    running(d, "v1")


def test_crashing_update_rolls_back(device, bundles):
    d = device
    running(d, "v1", confirmed=True)
    n = len(d.boots)
    code, body = push(d, bundles["crash"])
    assert code == 200  # it is a valid bundle; it just doesn't work
    wait_for(lambda: len(d.boots) >= n + 2, 60, what="reboot into the update and back")
    st = running(d, "v1", timeout=60)
    bs = d.boot_state()
    assert bs["active"] == "a" and bs["bad"] == "b" and not bs["trial"]
    assert any("rolled back b(crash) -> a(v1)" in e for e in bs["events"]), bs
    assert "rolled back" in st["boot"]["status"]
    assert "broken build" in (d.root / "crash.txt").read_text()


def test_update_that_never_gets_healthy_rolls_back(make_device, bundles):
    d = make_device(bundles["v1"], TRIAL_DEADLINE_S=12)
    running(d, "v1", confirmed=True)
    code, _ = push(d, bundles["sick"])
    assert code == 200
    running(d, "sick")
    st = running(d, "v1", timeout=60)
    assert "rolled back" in st["boot"]["status"]
    assert "not healthy" in (st["boot"]["reset_reason"] or ""), st["boot"]
    assert d.boot_state()["bad"] == "b"


def test_signed_bundles_required_when_key_configured(make_device, bundles):
    d = make_device(bundles["v2-signed"], UPDATE_KEY=SIGNING_KEY)
    running(d, "v2-signed", confirmed=True)
    code, body = push(d, bundles["v2"])
    assert code == 400 and "not signed" in body["error"]
    d2 = make_device(bundles["v1"], UPDATE_KEY="another-key")
    running(d2, "v1", confirmed=True)
    code, body = push(d2, bundles["v2-signed"])
    assert code == 400 and "bad signature" in body["error"]


def test_safe_mode_accepts_a_fresh_bundle(make_device, bundles):
    d = make_device(bundles["v1"])
    running(d, "v1", confirmed=True)
    push(d, bundles["v2"])
    running(d, "v2", confirmed=True)
    d.stop()
    # Break both slots so neither can boot.
    for slot in ("a", "b"):
        for f in (d.root / "fw" / slot / "app").glob("runtime.*"):
            f.write_bytes(b"M\x06\x00\x1fgarbage")
    d.boots.clear()
    d.start()
    st = d.wait_http(120, pred=lambda s: s.get("mode") == "safe")
    assert "boots without reaching health" in st["reason"]
    assert len(d.boots) >= 3
    code, body = curl(d.url("/update"), "-X", "POST", "--data-binary", f"@{bundles['v1']}")
    assert code == 401
    code, body = push(d, bundles["v1"])
    assert code == 200 and body["installed"] == "v1"
    running(d, "v1", timeout=60, confirmed=True)
    assert d.boot_state()["safe"] is None


def test_legacy_single_file_update_is_refused(device):
    code, body = curl("-X", "POST", "--data-binary", "x=1", "-H", f"Authorization: Bearer {device.token}",
                      device.url("/update?file=main.py"))
    assert code == 400 and "bundle" in body["error"]


class FakeReleases:
    """Serves bundles like GitHub Releases: a redirect to the asset host."""

    def __init__(self, files):
        import http.server
        import threading

        self.files = files  # {(channel, asset): bytes}
        outer = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                parts = self.path.strip("/").split("/")
                if parts[0] == "download" and len(parts) == 3:
                    self.send_response(302)
                    self.send_header("Location", f"/assets/{parts[1]}/{parts[2]}")
                    self.end_headers()
                    return
                data = outer.files.get(tuple(parts[1:3])) if parts[0] == "assets" else None
                if data is None:
                    self.send_response(404)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def url(self):
        return f"http://127.0.0.1:{self.port}/download/{{channel}}/{{asset}}"

    def close(self):
        self.srv.shutdown()


def test_self_update_from_releases(make_device, bundles):
    signed = bundles["v2-signed"]
    files = {
        ("main", "tree-firmware.bin"): signed.read_bytes(),
        ("main", "manifest.json"): (signed.parent / "manifest.json").read_bytes(),
        ("stable", "tree-firmware.bin"): bundles["v2"].read_bytes(),  # unsigned
        ("stable", "manifest.json"): (bundles["v2"].parent / "manifest.json").read_bytes(),
    }
    rel = FakeReleases(files)
    try:
        d = make_device(bundles["v1"], UPDATE_KEY=SIGNING_KEY, UPDATE_URL=rel.url())
        running(d, "v1", confirmed=True)
        code, res = curl("-H", f"Authorization: Bearer {d.token}", d.url("/update/github/check?channel=main"))
        assert code == 200 and res["latest"] == "v2-signed" and res["available"], res
        # An unsigned release is refused even though it is otherwise valid.
        code, res = curl("-X", "POST", "-H", f"Authorization: Bearer {d.token}", d.url("/update/github?channel=stable"))
        assert code == 202
        st = wait_for(lambda: (lambda s: s if s["update"]["state"] == "error" else None)(d.get("/")[1]), 20)
        assert "not signed" in st["update"]["error"]
        assert curl("-X", "POST", "-H", f"Authorization: Bearer {d.token}",
                    d.url("/update/github?channel=nope"))[0] == 409
        code, res = curl("-X", "POST", "-H", f"Authorization: Bearer {d.token}", d.url("/update/github?channel=main"))
        assert code == 202
        st = running(d, "v2-signed", timeout=60, confirmed=True)
        assert st["boot"]["previous"] == "a"
    finally:
        rel.close()


def test_self_update_needs_a_key(device):
    code, res = curl("-X", "POST", "-H", f"Authorization: Bearer {device.token}", device.url("/update/github"))
    assert code == 409 and "UPDATE_KEY" in res["error"]
