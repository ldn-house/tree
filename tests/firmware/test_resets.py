"""Deliberate resets must not look like watchdog hangs (found on hardware).

On the rp2 port machine.reset() is watchdog_reboot(), so reset_cause() says
WDT_RESET after any deliberate reset (mpremote's reset, flash.py, /reboot,
post-update reboots). The watchdog REASON register tells them apart: TIMER
for a genuine expiry, FORCE for a forced reboot. The simulator models that
(mocks/machine.py: REASON and persistent scratch registers)."""

from devsim import wait_for


FRESH = b"""
NAME = "fresh"
DURATION_MS = 60000
class Effect:
    def __init__(self, ctx):
        pass
    def render(self, t, frame):
        for i in range(len(frame)):
            frame[i] = 20
"""


def status(d):
    return d.get("/")[1]


def settled(d, effect):
    """The effect choice is saved (debounced) and the effect is past its
    probe window (fx_probe.txt removed): only watchdog scratch 0 names it."""
    wait_for(lambda: d.saved_state().get("effect") == effect and not (d.root / "fx_probe.txt").exists(), 15,
             what="%s saved and past its probe window" % effect)


def test_mpremote_interrupt_and_reset_does_not_quarantine(device):
    d = device
    d.post("/state", {"effect": "chase"})
    wait_for(lambda: status(d)["effect"]["running"] == "chase", 10)
    settled(d, "chase")  # scratch 0 names it
    # Switch to an effect that has never run and interrupt at once: it is in
    # its probe window, the state the hardware report hit.
    code, _ = d.request("PUT", "/effects/fresh", body=FRESH)
    assert code == 200
    d.post("/state", {"effect": "fresh"})
    wait_for(lambda: status(d)["effect"]["running"] == "fresh", 5)
    d.interrupt()
    wait_for(lambda: len(d.boots) >= 1, 20, what="reset")
    assert d.boots[0] == "machine.reset()", d.console()
    st = d.wait_http(30)
    assert st["effect"]["quarantined"] == [], st["effect"]
    assert st["boot"]["reset_cause"] == "reboot", st["boot"]
    # The choice was saved on the way out, and the effect simply runs again.
    st = wait_for(lambda: (lambda s: s if s["effect"]["running"] else None)(status(d)), 10)
    assert st["effect"]["running"] == "fresh" and st["state"]["effect"] == "fresh"


def test_reboot_endpoint_reports_a_deliberate_reset(device):
    d = device
    d.post("/state", {"effect": "solid"})
    wait_for(lambda: status(d)["effect"]["running"] == "solid", 10)
    settled(d, "solid")
    d.post("/reboot")
    wait_for(lambda: len(d.boots) >= 1, 20, what="reset")
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "reboot" and "requested" in st["boot"]["reset_reason"]
    assert st["effect"]["quarantined"] == []


LATE_HANG = b"""
NAME = "latehang"
DURATION_MS = 60000
class Effect:
    def __init__(self, ctx):
        pass
    def render(self, t, frame):
        if t > 4000:  # long after the 30-frame probe window
            while True:
                pass
"""


def test_genuine_late_hang_is_quarantined_via_scratch(device):
    """Past the probe window only watchdog scratch 0 names the effect; the
    simulator now models it (and REASON TIMER for the expiry)."""
    d = device
    code, _ = d.request("PUT", "/effects/latehang?activate=1", body=LATE_HANG)
    assert code == 200
    settled(d, "latehang")  # saved, and past the 30-frame probe window before it hangs at 4 s
    wait_for(lambda: len(d.boots) >= 1, 40, what="watchdog reset")
    assert d.boots[0] == "watchdog reset"
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "watchdog"
    assert st["boot"]["hang"]["phase"] == "effect render", st["boot"]
    assert "latehang" in st["effect"]["quarantined"], st["effect"]
    assert "watchdog fired" in st["effect"]["failed"]["latehang"]
    assert st["state"]["effect"] == "latehang" and st["effect"]["running"] != "latehang"


SLOW_START = b"""
NAME = "slowstart"
DURATION_MS = 60000
class Effect:
    def __init__(self, ctx):
        while True:
            pass
    def render(self, t, frame):
        pass
"""


def test_watchdog_hang_reports_where_the_loop_was(device):
    """The tree reset with reset_cause "watchdog" and no reason: the loop was
    blocked, and nothing in RAM survived to say where. app.trace keeps the
    phase in watchdog scratch 1, which does."""
    d = device
    try:  # validation constructs it inside the request, which never answers
        d.request("PUT", "/effects/slowstart", body=SLOW_START)
    except OSError:
        pass
    wait_for(lambda: len(d.boots) >= 1, 40, what="watchdog reset")
    assert d.boots[0] == "watchdog reset"
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "watchdog"
    assert st["boot"]["hang"]["phase"] == "effect start", st["boot"]
    assert st["boot"]["reset_reason"].startswith("hung in effect start")
    assert any("watchdog expired: hung in effect start" in line for line in d.get("/log")[1]["lines"])
    # A deliberate reset clears it.
    d.post("/reboot")
    wait_for(lambda: len(d.boots) >= 2, 20, what="reboot")
    st = d.wait_http(30)
    assert st["boot"]["reset_cause"] == "reboot" and st["boot"]["hang"] is None
