"""Board-sized heap campaign (see heapcheck.py), first half: three auto-cycle
loops through the repo's effects with MQTT connected, then the TLS update
check and self-update on the heap those loops left behind. The second half
is test_heap_updated.py. Needs a 32-bit MicroPython unix build in
$MICROPYTHON32 (CI builds one with MICROPY_FORCE_32BIT=1)."""


def test_board_sized_heap_cycles_then_tls_self_update(heap_campaign):
    res = heap_campaign("--before", "3", "--after", "0")
    assert {"cycle_all_effects", "tls_check", "tls_self_update"} <= set(res["steps"])
