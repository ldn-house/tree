"""Commands reach the LEDs promptly: from the command to the simulated wire.

Coarse on purpose. The unix port isn't the Pico and CI runners are noisy, so
this doesn't time milliseconds: it catches structural delays, such as the 5 s
lead HA's calibration button once asked for, a command waiting on a slow poll
or on a publishing loop. Commands take about 30-110 ms here, so LIMIT_S is ten
times that. Each scenario gets TRIES attempts and the best must be under the
limit: a structural delay shows on every attempt, a stalled runner doesn't.
Ramps and blends are off (transition 0, style none), so the first frame after
a command is the whole change; with them on, that frame is the start of a fade.
"""
import base64
import json
import time

from devsim import wait_for
from test_mqtt import mdev  # noqa: F401  (MQTT device fixture)

LIMIT_S = 1.0
TRIES = 3
FULL = 47000  # solid white under the 1.5 A cap: 47812


def frame(d):
    return base64.b64decode(d.get("/frame")[1]["frame"])


def level(d):
    return sum(frame(d))


def best(what, prepare, act, done):
    """Seconds from act() until done(), best of up to TRIES attempts."""
    times = []
    for _ in range(TRIES):
        prepare()
        t0 = time.monotonic()
        act()
        wait_for(done, 10, interval=0.01, what=what)
        times.append(time.monotonic() - t0)
        if times[-1] < LIMIT_S:
            break
    print("%s: %s s" % (what, ", ".join("%.3f" % t for t in times)))
    assert min(times) < LIMIT_S, "%s took %s s" % (what, times)


def test_commands_reach_the_leds_promptly(mdev, bus):  # noqa: F811
    d = mdev

    def user(**values):
        assert d.post("/state", dict(values, transition=0))[0] == 200

    def full_white():
        user(power=True, effect="solid", brightness=255)
        wait_for(lambda: level(d) > FULL, 5, interval=0.01, what="full white")

    def light(**cmd):
        bus.pub("xmas_tree/set", json.dumps(dict(cmd, transition=0)))

    user(power=True, effect="solid", brightness=255, colour_accuracy=0, transition_style="none")
    dim = lambda: level(d) < 30000  # noqa: E731  (brightness 128: about half)

    best("brightness, HA", full_white, lambda: light(state="ON", brightness=128), dim)
    best("brightness, POST /state", full_white, lambda: user(brightness=128), dim)
    # The fixture's height effect has no red: the cut is complete when red is.
    best("effect, HA", full_white, lambda: light(effect="height"), lambda: sum(frame(d)[0::3]) == 0)
    best("power off, HA", full_white, lambda: light(state="OFF"), lambda: level(d) == 0)

    def dark():
        user(power=False)
        wait_for(lambda: level(d) == 0, 5, interval=0.01, what="dark")

    best("power on, HA", dark, lambda: light(state="ON"), lambda: level(d) > 0)

    # HA shows the change once the state comes back: the light's state topic.
    t0 = [0]

    def ha_dim():
        t0[0] = time.time()
        light(state="ON", brightness=128)

    best("state back to HA", full_white, ha_dim,
         lambda: any(json.loads(p).get("brightness") == 128 for _t, p in bus.since(t0[0], "xmas_tree/state")))

    # Calibration lock ("start calibration"): the effect stops, all off.
    def unlocked():
        d.post("/calibrate/stop")
        full_white()

    best("calibration lock, POST /calibrate/start", unlocked, lambda: d.post("/calibrate/start"),
         lambda: level(d) == 0)

    # HA's "Play calibration sequence" button, sending exactly what discovery
    # gives HA. Timed playback bypasses /frame (#53), so its first latched
    # frame is read from the playback log; play() replaces the log in the
    # same step that makes the state "playing", so "playing" means the log
    # is this attempt's.
    cfg = wait_for(lambda: bus.last_json("homeassistant/button/xmas_tree/calibrate_play/config"), 20,
                   what="calibration button discovery")
    stop = wait_for(lambda: bus.last_json("homeassistant/button/xmas_tree/calibrate_stop/config"), 20,
                    what="stop button discovery")

    def idle():
        bus.pub(stop["command_topic"], stop["payload_press"])
        wait_for(lambda: d.get("/calibrate")[1]["state"] == "idle", 10, what="calibration idle")
        full_white()

    def first_latch():
        if d.get("/calibrate")[1]["state"] != "playing":
            return False
        return d.get("/calibrate/log")[1]["frames_shown"] >= 1

    best("calibration play, HA button", idle, lambda: bus.pub(cfg["command_topic"], cfg["payload_press"]),
         first_latch)
    bus.pub(stop["command_topic"], stop["payload_press"])
    assert len(d.boots) == 0
