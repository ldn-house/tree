"""Device-level regressions for review round 1 (fw21-review-round1.md).
Each test reproduces the reviewed failure on the simulated board."""

import hashlib
import hmac
import http.server
import json
import socket
import subprocess
import threading
import time

import pytest

import devsim
from conftest import SIGNING_KEY, curl
from devsim import wait_for
from test_updates import FakeReleases, push, running


def status(d):
    return d.get("/")[1]


def sign(kind, body, key=SIGNING_KEY):
    return hmac.new(key.encode(), b"tree-" + kind.encode() + b"\n" + body, hashlib.sha256).hexdigest()


@pytest.fixture(scope="module")
def gens(tmp_path_factory):
    """Signed bundles with fixed generations, for replay tests."""
    tmp = tmp_path_factory.mktemp("gens")
    return {
        g: devsim.build_bundle(tmp / str(g), key=SIGNING_KEY, version="g%d" % g, generation=g)
        for g in (1000, 2000)
    }


# -- B1: the boot can't be recorded ----------------------------------------------


def test_b1_unwritable_boot_state_goes_to_safe_mode_and_recovers(make_device, bundles):
    d = make_device(bundles["v1"])
    running(d, "v1", confirmed=True)
    d.stop()
    # boot.json.tmp as a directory: every boot-state write now fails, like a
    # full or failing flash would.
    (d.root / "boot.json.tmp").mkdir()
    (d.root / "boot.json.tmp" / "x").write_text("x")
    d.boots.clear()
    d.start()
    st = d.wait_http(30, pred=lambda s: s.get("mode") == "safe")
    assert "cannot record the boot" in st["reason"]
    assert len(d.boots) == 0  # straight to recovery, no failing app loop
    # Safe mode reclaimed the stray temp tree, so an upload can commit.
    assert not (d.root / "boot.json.tmp").exists()
    code, body = push(d, bundles["v2"])
    assert code == 200 and body["installed"] == "v2", body
    running(d, "v2", timeout=60, confirmed=True)


# -- H3: rejected uploads and dry-runs keep the fallback ---------------------------


def test_h3_fallback_survives_rejected_upload_and_trial_dry_run(make_device, bundles, tmp_path):
    d = make_device(bundles["v1"], TRIAL_HEALTHY_S=5)  # time to act during v2's trial
    d.wait_http(40, pred=lambda s: not s["boot"]["trial"])
    code, _ = push(d, bundles["v2"])
    assert code == 200
    st = running(d, "v2")
    assert st["boot"]["trial"] and st["boot"]["previous"] == "a"
    a = (d.root / "fw" / "a" / "manifest.json").read_bytes()
    # During the trial: a dry-run verifies without touching slot a...
    code, body = push(d, bundles["v1"], path="/update?dry_run=1&allow_downgrade=1")
    assert code == 200 and body["verified"] == "v1", body
    # ...a real install is refused before anything is staged...
    code, body = push(d, bundles["v1"], path="/update?allow_downgrade=1")
    assert code == 409 and "not confirmed" in body["error"]
    # ...and a malformed upload is rejected without touching either slot.
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"NOTFW!!!" + bytes(200))
    code, body = push(d, bad)
    assert code in (400, 409), body
    assert (d.root / "fw" / "a" / "manifest.json").read_bytes() == a
    assert not (d.root / "fw" / "new").exists()
    # The rollback target is still there: a failed trial goes back to v1.
    d.kill_now("pwron")
    st = running(d, "v1", timeout=60)
    assert "rolled back" in st["boot"]["status"]


# -- H4 (b): effect code needs the signing key ---------------------------------------

EVIL = b"""with open('boot.py', 'w') as f:
    f.write('# replaced by an effect\\n')
NAME = 'evil'
class Effect:
    def __init__(self, ctx): pass
    def render(self, t, frame): pass
"""

GOOD = b"""NAME = 'blue'
DURATION_MS = 3000
class Effect:
    def __init__(self, ctx):
        self.n = ctx.n
    def render(self, t, frame):
        for i in range(0, len(frame), 3):
            frame[i + 2] = 50
"""


def test_h4_unsigned_effects_are_refused_before_they_run(make_device, bundles):
    d = make_device(bundles["v2-signed"], UPDATE_KEY=SIGNING_KEY)
    running(d, "v2-signed", confirmed=True)
    before = (d.root / "boot.py").read_bytes()
    code, body = d.request("PUT", "/effects/evil", body=EVIL)
    assert code == 403 and "signed" in body["error"]
    code, body = d.request("PUT", "/effects/evil", body=EVIL, headers={"X-Signature": "00" * 32})
    assert code == 403
    assert (d.root / "boot.py").read_bytes() == before  # the code never ran
    body = GOOD
    code, res = d.request("PUT", "/effects/blue?activate=1", body=body,
                          headers={"X-Signature": sign("effect", b"blue\n" + body)})
    assert code == 200 and res["activated"], res
    # A signature for another name doesn't transfer.
    code, _ = d.request("PUT", "/effects/other", body=body, headers={"X-Signature": sign("effect", b"blue\n" + body)})
    assert code == 403


# -- H5: replay and downgrade -------------------------------------------------------


def test_h5_replayed_older_release_is_refused(make_device, gens):
    old, new = gens[1000], gens[2000]
    rel = FakeReleases({("main", "tree-firmware.bin"): old.read_bytes(),
                        ("main", "manifest.json"): (old.parent / "manifest.json").read_bytes()})
    try:
        d = make_device(new, UPDATE_KEY=SIGNING_KEY, UPDATE_URL=rel.url())
        running(d, "g2000", confirmed=True)
        assert d.boot_state()["floor"] == 2000
        # An attacker on the network serves the old, correctly signed bundle.
        code, _ = curl("-X", "POST", "-H", f"Authorization: Bearer {d.token}", d.url("/update/github?channel=main"))
        assert code == 202
        st = wait_for(lambda: (lambda s: s if s["update"]["state"] == "error" else None)(status(d)), 30)
        assert "older than the minimum" in st["update"]["error"]
        assert not (d.root / "fw" / "new").exists() and (d.root / "fw" / "a" / "manifest.json").exists()
        # The same over HTTP without asking: refused. Asking explicitly: allowed.
        code, body = push(d, old)
        assert code == 400 and "older" in body["error"]
        code, body = push(d, old, path="/update?allow_downgrade=1")
        assert code == 200 and body["installed"] == "g1000"
        st = running(d, "g1000", timeout=60, confirmed=True)
        assert d.boot_state()["floor"] == 2000  # a deliberate downgrade doesn't lower the floor
    finally:
        rel.close()


# -- H6: unbounded request lines --------------------------------------------------


def test_h6_unterminated_request_line_is_cut_off(device):
    d = device
    heap0 = status(d)["free_memory"]
    sent = 0
    s = socket.create_connection(("127.0.0.1", d.port))
    s.settimeout(5)
    reply = b""
    try:
        for _ in range(300):  # up to 300 KB, no newline
            s.sendall(b"G" * 1024)
            sent += 1024
    except OSError:
        pass  # the server closed on us
    try:
        reply = s.recv(4096)
    except OSError:
        pass
    s.close()
    assert sent < 300 * 1024 or b"431" in reply, (sent, reply[:80])
    assert b"431" in reply or reply == b""
    st = status(d)
    assert st["free_memory"] > heap0 - 64 * 1024, (heap0, st["free_memory"])
    assert len(d.boots) == 0


class Trickle:
    """A release server that sends its headers one byte a second, forever,
    or an endless header line, for the outbound-client checks."""

    def __init__(self, mode):
        self.srv = socket.socket()
        self.srv.bind(("127.0.0.1", 0))
        self.srv.listen(5)
        self.port = self.srv.getsockname()[1]
        self.mode = mode
        self.stop = False
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        self.srv.settimeout(0.5)
        while not self.stop:
            try:
                c, _ = self.srv.accept()
            except OSError:
                continue
            threading.Thread(target=self._client, args=(c,), daemon=True).start()

    def _client(self, c):
        try:
            c.recv(4096)
            if self.mode == "trickle":
                c.sendall(b"HTTP/1.0 200 OK\r\n")
                while not self.stop:
                    c.sendall(b"X-Slow: y\r\n")  # a header every second, never done
                    time.sleep(1)
            else:
                c.sendall(b"HTTP/1.0 200 OK\r\nX-Big: ")
                while not self.stop:
                    c.sendall(b"A" * 4096)
        except OSError:
            pass
        finally:
            c.close()

    def url(self):
        return f"http://127.0.0.1:{self.port}/{{channel}}/{{asset}}"

    def close(self):
        self.stop = True
        self.srv.close()


def test_h6_outbound_headers_are_bounded(make_device, bundles):
    t = Trickle("flood")
    try:
        d = make_device(bundles["v1"], UPDATE_URL=t.url())
        running(d, "v1", confirmed=True)
        code, res = curl("-H", f"Authorization: Bearer {d.token}", d.url("/update/github/check"))
        assert code == 200 and "too large" in res["error"], res
        assert len(d.boots) == 0
    finally:
        t.close()


# -- H7: a slow update check never resets healthy firmware -------------------------


def test_h7_slow_update_check_does_not_reset(make_device, bundles):
    t = Trickle("trickle")
    try:
        d = make_device(bundles["v1"], UPDATE_URL=t.url(), UPDATE_CHECK_HOURS=1, UPDATE_CHECK_FIRST_S=3)
        running(d, "v1", confirmed=True)
        # The check starts at 3 s and the server never finishes its headers.
        st = wait_for(lambda: (lambda s: s if s["update"]["check_error"] else None)(status(d)), 70,
                      what="bounded check")
        assert "timed out" in st["update"]["check_error"], st["update"]
        assert len(d.boots) == 0 and status(d)["supervisor"]["problems"] == []
    finally:
        t.close()


# -- M9: a broken config.py is recoverable over Wi-Fi ---------------------------------


def test_m9_broken_config_recovers_over_wifi(make_device, bundles):
    d = make_device(bundles["v2-signed"], UPDATE_KEY=SIGNING_KEY)
    running(d, "v2-signed", confirmed=True)
    assert (d.root / "config.good.json").exists()
    d.stop()
    (d.root / "config.py").write_text("WIFI_SSID = 'testnet'\nthis line is not python\n")
    d.boots.clear()
    d.start()
    st = d.wait_http(40)
    assert st["config_source"] == "config.good.json" and "failed to load" in st["config_error"]
    assert st["wifi"]["state"] == "up" or wait_for(lambda: status(d)["wifi"]["state"] == "up", 30)
    values = dict(d.config, HTTP_PORT=d.port)
    body = json.dumps(values).encode()
    code, res = d.request("PUT", "/config", body=body)  # unsigned: refused
    assert code == 403
    code, res = d.request("PUT", "/config", body=body, headers={"X-Signature": sign("config", body)})
    assert code == 200 and "HTTP_TOKEN" in res["written"], res
    st = d.wait_http(40, pred=lambda s: s["config_error"] is None and len(d.boots) >= 1)
    assert st["config_source"] == "config.py"
    compile((d.root / "config.py").read_text(), "config.py", "exec")


# -- M10: slots held by unauthenticated slow clients --------------------------------


def test_m10_slow_unauthenticated_uploads_cannot_hold_the_server(device):
    d = device
    socks = []
    for _ in range(4):
        s = socket.create_connection(("127.0.0.1", d.port))
        s.sendall(b"POST /update HTTP/1.0\r\nContent-Length: 524288\r\n\r\n")
        socks.append(s)

    stop = threading.Event()

    def dribble():
        while not stop.is_set():
            for s in socks:
                try:
                    s.send(b"x")
                except OSError:
                    pass
            time.sleep(0.5)

    th = threading.Thread(target=dribble, daemon=True)
    th.start()
    try:
        time.sleep(5)  # the anonymous drain budget is 3 s
        t0 = time.time()
        code, st = d.get("/", timeout=5)
        assert code == 200 and time.time() - t0 < 3
        code, _ = push(d, devsim.ROOT / "README.md")  # an authenticated request also gets through
        assert code == 400
    finally:
        stop.set()
        for s in socks:
            s.close()


def test_m10_safe_mode_slow_client_is_cut_off_without_reset(make_device, bundles):
    d = make_device(bundles["v1"])
    running(d, "v1", confirmed=True)
    d.stop()
    st = d.boot_state()
    st["safe"] = "test"
    (d.root / "boot.json").write_text(json.dumps(st))
    d.boots.clear()
    d.start()
    d.wait_http(30, pred=lambda s: s.get("mode") == "safe")
    s = socket.create_connection(("127.0.0.1", d.port))
    t0 = time.time()
    closed = False
    try:
        while time.time() - t0 < 20:
            s.sendall(b"P")  # a header byte every 1.5 s, never finishing
            time.sleep(1.5)
    except OSError:
        closed = True
    s.close()
    assert closed or time.time() - t0 >= 20
    assert len(d.boots) == 0  # the watchdog never fired
    assert d.get("/", timeout=5)[1]["mode"] == "safe"


# -- clock boundaries -----------------------------------------------------------------


def test_long_finite_playback_is_not_cut_by_ttl(device):
    import sys

    sys.path.insert(0, str(devsim.ROOT / "firmware" / "app"))
    import seqfile

    d = device
    lit = [i % 2 == 0 for i in range(500)]
    data = seqfile.encode([seqfile.Frame(40, (20, 20, 20), (seqfile.pack_plane(lit, 500),))], 500, 1)
    assert d.post("/calibrate/sequence", data)[0] == 200
    code, res = d.post("/calibrate/play?sequence=uploaded&loops=65535&ttl_s=2")
    assert code == 200 and res["loops"] == 65535
    time.sleep(4)
    st = d.get("/calibrate")[1]
    assert st["state"] == "playing" and st["playing"]["t0_ms_ago"] > 2000
    d.post("/calibrate/stop")
