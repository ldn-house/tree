"""Board-sized heap campaign (see heapcheck.py), second half: the TLS
self-update straight after boot, then three auto-cycle loops and the default
calibration sequence on the updated firmware, with MQTT connected. The first
half is test_heap.py."""

import devsim


def test_board_sized_heap_after_tls_self_update(heap_campaign):
    res = heap_campaign("--before", "0", "--after", "3", "--calib", str(devsim.ROOT / "sequences" / "default.tcal"))
    assert {"tls_self_update", "cycle_after_update", "calibration_default"} <= set(res["steps"])
