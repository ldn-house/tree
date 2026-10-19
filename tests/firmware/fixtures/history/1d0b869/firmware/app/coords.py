"""Per-LED coordinates for effects (docs/effect-api.md, "Coordinates").

/coords.txt (pushed by the calibration tools, already in the contract frame)
wins; otherwise a legacy coords_compact.txt (pushed to /, or bundled with the
firmware) is rescaled by effects._coords. A missing, short or malformed file never stops
the firmware: missing LEDs are None and effects that need coordinates are
skipped when there are none at all (review H3).
"""

from app.log import log

USER_FILE = "coords.txt"
LEGACY_FILE = "coords_compact.txt"


def load(n, slot_dir):
    try:
        from effects import _coords
    except Exception as e:
        log("coords: effects._coords unavailable (%r)" % (e,))
        return None
    for path, legacy in ((USER_FILE, False), (LEGACY_FILE, True), (slot_dir + "/data/coords_compact.txt", True)):
        try:
            c = _coords.load(path, n, legacy=legacy)
        except Exception as e:
            log("coords: %s is malformed (%r)" % (path, e))
            continue
        if c is None:
            continue
        known = sum(1 for p in c if p is not None)
        if known == 0:
            log("coords: %s has no positions" % path)
            continue
        log("coords: %d/%d LEDs from %s" % (known, n, path))
        return c
    log("coords: none loaded; 3D effects disabled")
    return None


def count(coords):
    return 0 if not coords else sum(1 for p in coords if p is not None)
