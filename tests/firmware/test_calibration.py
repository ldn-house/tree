"""Calibration protocol (docs/calibration-protocol.md) on the simulated device."""

import base64
import json
import shutil
import subprocess
import sys
import time

import pytest

import devsim
from conftest import curl
from devsim import wait_for

sys.path.insert(0, str(devsim.ROOT / "firmware" / "app"))
import seqfile  # noqa: E402  (the vendored, CPython-compatible codec)

N = 500


def frame(d):
    return base64.b64decode(d.get("/frame")[1]["frame"])


def cal(d):
    return d.get("/calibrate")[1]


def seq(frames, loops=1):
    return seqfile.encode(frames, N, loops)


def mono(ms, level, lit):
    return seqfile.Frame(ms, (level, level, level), (seqfile.pack_plane(lit, N),))


def test_cors_and_auth(device):
    d = device
    r = subprocess.run(["curl", "-s", "-i", "-X", "OPTIONS", d.url("/calibrate/play")], capture_output=True, text=True)
    assert "204" in r.stdout.splitlines()[0]
    assert "Access-Control-Allow-Origin: *" in r.stdout
    assert "Authorization" in r.stdout and "X-Tree-Token" in r.stdout
    assert "Access-Control-Allow-Private-Network: true" in r.stdout
    r = subprocess.run(["curl", "-s", "-i", d.url("/calibrate")], capture_output=True, text=True)
    assert "Access-Control-Allow-Origin: *" in r.stdout
    assert curl(d.url("/calibrate/start"))[0] == 401
    assert curl("-H", f"X-Tree-Token: {d.token}", d.url("/calibrate/start"))[0] == 200


def test_legacy_endpoints_go_through_the_power_cap(device):
    d = device
    assert d.get("/calibrate/led?n=5")[0] == 401
    st = cal(d)
    assert st["state"] == "idle" and st["power_budget"] == 47812  # 1.5 A at 8 mA per channel and st["min_frame_ms"] == 20
    assert d.get("/calibrate/led?n=5", token=True)[0] == 400  # not locked: legacy error
    code, res = d.get("/calibrate/start", token=True)
    assert code == 200 and res["success"] and res["state"] == "locked"
    wait_for(lambda: sum(frame(d)) == 0, 5, what="all off")  # locked: effects stop
    code, res = d.get("/calibrate/led?n=5", token=True)
    assert res == {"success": True, "led": 5}
    f = wait_for(lambda: (lambda f: f if f[15:18] == bytes([255, 255, 255]) else None)(frame(d)), 5, what="LED 5")
    assert sum(f) == 765  # brightness forced to full
    assert d.get("/calibrate/status")[1] == {"calibration_mode": True, "current_led": 5, "total_leds": 500}
    code, res = d.get("/calibrate/binary?bit=3", token=True)
    assert res["leds_on"] == 248  # LEDs 0-499 with bit 3 set
    f = wait_for(lambda: (lambda f: f if f[3 * 8] > 0 else None)(frame(d)), 5, what="binary pattern")
    assert sum(f) <= 47812 and f[0] == 0  # 248 x 240 = 59520 raw, capped (review M6)
    assert d.get("/calibrate/clear", token=True)[0] == 200
    wait_for(lambda: sum(frame(d)) == 0, 5, what="cleared")
    assert d.get("/calibrate/stop", token=True)[1]["state"] == "idle"
    wait_for(lambda: d.get("/")[1]["effect"]["running"], 5)


def test_calibration_frames_ignore_the_brightness_floor(device):
    """Calibration frames are exact whatever the floor and brightness."""
    d = device
    d.post("/state", {"floor": 25, "brightness": 3})
    d.get("/calibrate/start", token=True)
    d.get("/calibrate/led?n=7", token=True)
    f = wait_for(lambda: (lambda f: f if f[21:24] == bytes([255, 255, 255]) else None)(frame(d)), 5, what="LED 7")
    assert sum(f) == 765
    lit = [i % 3 == 0 for i in range(N)]
    code, res = d.post("/calibrate/frame", seq([mono(1000, 40, lit)]))
    assert code == 200, res
    want = b"".join(bytes([40, 40, 40]) if x else bytes(3) for x in lit)
    wait_for(lambda: frame(d) == want, 5, what="static calibration frame, exact")
    d.get("/calibrate/stop", token=True)


def test_ttl_expiry_and_power_off_unlock(device):
    d = device
    d.post("/calibrate/start?ttl_s=2")
    assert cal(d)["state"] == "locked"
    time.sleep(1.5)
    assert cal(d)["state"] == "locked"  # GET /calibrate refreshed the TTL...
    # ...then it expired (review M7). Poll GET /, which (unlike /calibrate)
    # doesn't refresh the TTL.
    wait_for(lambda: d.get("/")[1]["calibration"] == "idle", 6, what="TTL expiry")
    wait_for(lambda: d.get("/")[1]["effect"]["running"], 5)
    # Brightness changes while locked are deferred; power off unlocks at once.
    d.post("/calibrate/start")
    code, res = d.post("/state", {"brightness": 33})
    assert code == 200 and cal(d)["state"] == "locked"
    d.post("/state", {"power": False})
    assert cal(d)["state"] == "idle"
    d.post("/state", {"power": True})
    st = wait_for(lambda: (lambda s: s if s["effect"]["running"] else None)(d.get("/")[1]), 5)
    assert st["state"]["brightness"] == 33


def test_sequence_upload_play_and_log(device):
    d = device
    lit = [i % 2 == 0 for i in range(N)]
    # 250 LEDs x 3 x 64 = 48000 is over the 1.5 A budget (47812): reported.
    code, res = d.post("/calibrate/sequence", seq([mono(40, 64, lit), mono(40, 60, lit)]))
    assert code == 200 and res["over_budget_frames"] == [0]
    frames = [mono(40, 60, lit), mono(40, 60, [not b for b in lit]), mono(40, 0, [0] * N)]
    data = seq(frames, loops=2)
    code, res = d.post("/calibrate/sequence", data)
    assert code == 200 and res["frames"] == 3 and res["loop_ms"] == 120 and res["over_budget_frames"] == []
    bad = bytearray(data)
    bad[30] ^= 1
    code, res = d.post("/calibrate/sequence", bytes(bad))
    assert code == 400 and "crc" in res["error"]
    code, res = d.post("/calibrate/sequence", seq([mono(5, 1, lit)]))
    assert code == 400 and "min_frame_ms" in res["error"]
    code, res = d.post("/calibrate/sequence", b"x" * 70000)
    assert code == 413

    code, res = d.post("/calibrate/play?sequence=uploaded&delay_ms=300")
    assert code == 200 and res == {"sequence": "uploaded", "loops": 2, "loop_ms": 120, "t0_in_ms": 300}  # delay above the 25 ms minimum lead
    assert cal(d)["state"] == "playing"
    assert d.post("/calibrate/play")[0] == 409
    assert d.post("/calibrate/sequence", data)[0] == 409
    wait_for(lambda: cal(d)["state"] == "locked", 5, what="playback end")
    log = d.get("/calibrate/log")[1]
    assert log["frames_shown"] == 6 and log["frames_skipped"] == 0 and log["capped_frames"] == 0
    # The latch error measures host scheduling here (hardware: ±4 ms, on
    # the checklist); what must hold is that no frame missed its slot.
    print("latch error (unix port):", log["late_us"])
    wait_for(lambda: sum(frame(d)) == 0, 5, what="all off at the end")
    # Implicit lock: released 2 s after playback ends.
    wait_for(lambda: cal(d)["state"] == "idle", 5, what="implicit unlock")


def test_frame_and_default_sequence(device, tmp_path):
    d = device
    one = seq([seqfile.Frame(1, (10, 20, 30), (seqfile.pack_plane([1] * N, N),) * 3)])
    assert d.post("/calibrate/frame", one)[0] == 409  # not locked
    d.post("/calibrate/start")
    code, res = d.post("/calibrate/frame", one)
    assert code == 200
    # Static frames are refreshed on the calibration task's 100 ms tick:
    # ~0.2 s to reach the wire here, so a fixed 0.3 s wait was too tight on CI.
    f = wait_for(lambda: (lambda f: f if f[:3] == bytes([10, 20, 30]) else None)(frame(d)), 5, what="static frame")
    assert sum(f) == 60 * N
    d.post("/calibrate/stop")
    # The bundle carries sequences/default.tcal as data/calib_default.tcal,
    # so the HA "Play calibration sequence" button works out of the box.
    want = (devsim.ROOT / "sequences" / "default.tcal").read_bytes()
    assert (d.root / "fw" / "a" / "data" / "calib_default.tcal").read_bytes() == want
    st = cal(d)
    assert st["sequences"]["default"] == {"crc32": "83cffe28", "frames": 57, "loop_ms": 6600}
    frames = st["sequences"]["default"]["frames"]
    # A frame is skipped only when it is overdue by its whole duration (100 ms
    # here), which the firmware avoids by deferring GC, the reserve build and
    # state saves during playback; a shared CI runner can still freeze the
    # device's process that long. So the sequence gets three tries to play
    # every frame: a stall in the firmware would skip frames every time.
    for attempt in range(3):
        code, res = d.post("/calibrate/play?sequence=default&loops=1")
        assert code == 200
        loop_ms = res["loop_ms"]
        t0 = time.time()
        # Poll for the end rather than sleeping loop_ms + 1 s: on a loaded CI
        # runner the test process and the device are scheduled independently.
        log = wait_for(lambda: (lambda l: l if l["frames_shown"] + l["frames_skipped"] >= frames else None)(
            d.get("/calibrate/log")[1]), loop_ms / 1000 + 20, interval=0.1, what="playback end")
        print("default sequence: %d frames in %.2f s after play (loop %.1f s), latch error (unix port) %s, worst %s,"
              " by frame %s" % (log["frames_shown"], time.time() - t0, loop_ms / 1000, log["late_us"],
                                log.get("worst_frame"), log.get("late_us_by_frame")))
        if log["frames_shown"] == frames and log["frames_skipped"] == 0:
            break
        wait_for(lambda: cal(d)["state"] != "playing", 10, what="playback stopped")
    assert log["frames_shown"] == frames and log["frames_skipped"] == 0


def test_calibration_over_mqtt(make_device, bundles, broker, bus):
    d = make_device(bundles["v1"], MQTT_BROKER="127.0.0.1", MQTT_PORT=broker.port, MQTT_KEEPALIVE=5)
    d.wait_http(30)
    bus.wait("xmas_tree/availability", lambda p: p == b"online", 30)
    lit = [i % 3 == 0 for i in range(N)]
    d.post("/calibrate/sequence", seq([mono(100, 40, lit), mono(100, 0, lit)], loops=0))
    t0 = time.time()
    bus.pub("xmas_tree/calibrate/set", '{"action":"play","sequence":"uploaded","ttl_s":60}')
    bus.wait("xmas_tree/calibrate/state", lambda p: json.loads(p)["state"] == "playing", 10, after=t0)
    wait_for(lambda: len(bus.since(t0, "xmas_tree/calibrate/state")) >= 3, 10, what="state once a second while playing")
    bus.pub("xmas_tree/calibrate/set", '{"action":"bogus"}')
    bus.pub("xmas_tree/calibrate/set", "nope")
    # HA power off always unlocks.
    bus.pub("xmas_tree/set", '{"state":"OFF"}')
    bus.wait("xmas_tree/calibrate/state", lambda p: json.loads(p)["state"] == "idle", 10, after=t0)
    assert d.get("/")[1]["state"]["power"] is False
    assert len(d.boots) == 0


def test_missing_default_sequence_is_a_clear_409(device):
    d = device
    (d.root / "fw" / "a" / "data" / "calib_default.tcal").unlink()
    code, res = d.post("/calibrate/play?sequence=default")
    assert code == 409 and res["error"] == "no default sequence"


def test_settings_are_not_saved_to_flash_during_playback(device):
    """A flash write can stall the Pico for tens of ms (an erase stops all
    code), and playback frames are timed to the millisecond, so a settings
    change made during playback is saved when it ends, not during it."""
    d = device
    lit = [i % 2 == 0 for i in range(N)]
    d.post("/calibrate/sequence", seq([mono(100, 30, lit), mono(100, 0, lit)], loops=0))
    code, _ = d.post("/calibrate/play?sequence=uploaded&ttl_s=60")
    assert code == 200
    before = d.saved_state().get("speed")
    assert d.post("/state", {"speed": 2.5})[0] == 200
    assert d.get("/")[1]["state"]["speed"] == 2.5  # applied at once...
    time.sleep(3.5)  # ...past the 2 s save debounce (an absence check: nothing may be written)
    assert d.saved_state().get("speed") == before, "saved during playback"
    d.post("/calibrate/halt")
    wait_for(lambda: d.saved_state().get("speed") == 2.5, 10, what="saved after playback")
    d.post("/calibrate/stop")


def test_infinite_playback_from_idle_runs_until_its_ttl(device):
    """HA's "Play calibration sequence" button plays loops=0 with ttl_s from
    idle. The implicit lock left the expiry at "now", so the playback was
    unlocked by the calibration task's next check, within ~100 ms."""
    d = device
    lit = [i % 2 == 0 for i in range(N)]
    d.post("/calibrate/sequence", seq([mono(100, 30, lit), mono(100, 0, lit)], loops=0))
    t0 = time.time()
    code, res = d.post("/calibrate/play?sequence=uploaded&ttl_s=3")
    assert code == 200 and res["loops"] == 0
    time.sleep(1.5)  # an absence check: it must not have stopped
    assert d.get("/")[1]["calibration"] == "playing"
    wait_for(lambda: d.get("/")[1]["calibration"] == "idle", 10, what="TTL end")
    assert time.time() - t0 >= 2.5  # ended by its 3 s TTL, not at once
    assert "ttl expired" in d.console()
