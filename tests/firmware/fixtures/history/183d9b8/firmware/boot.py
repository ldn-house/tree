# Recovery layer: installed over USB only, never written by OTA.
#
# boot.py runs on every reset, including the soft resets mpremote performs
# before raw-REPL commands (those skip main.py). It only keeps a watchdog that
# is already running fed, so a USB session is never cut short. The boot
# counter and slot selection live in main.py, which runs on real boots only.
try:
    import bootmgr

    bootmgr.early()
except BaseException as e:  # never let boot.py stop the board booting
    try:
        import sys

        sys.print_exception(e)
    except Exception:
        pass
