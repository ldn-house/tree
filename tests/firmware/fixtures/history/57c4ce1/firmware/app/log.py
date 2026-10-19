"""Console logging plus a small ring buffer exposed at GET /log."""

import time

_lines = []
MAX_LINES = 40


def log(*args):
    msg = " ".join(str(a) for a in args)
    print(msg)
    line = "%d %s" % (time.ticks_ms() // 1000, msg)
    _lines.append(line)
    if len(_lines) > MAX_LINES:
        del _lines[0]


def lines():
    return list(_lines)
