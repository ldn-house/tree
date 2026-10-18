"""Coordinate loading for the runtime, renderer and viewer. MicroPython-safe.

Effects receive coordinates in the contract frame (docs/effect-api.md):
y up, y = 0 at the lowest LED and 1 at the highest; x and z in the same units
with the origin on the trunk axis; +x is the front camera's image right.

Two file formats, both one line per LED, "x,y,z" or "null":
- coords.txt: already in the contract frame (produced by the calibration
  pipeline); loaded as-is.
- coords_compact.txt: the legacy triangulate.py output. Its x/z are
  normalised by image width and y by image height, which stretches the tree
  horizontally by 1080/608 (review H8), and its origin is the bounding-box
  centre. adapt_legacy() rescales it into the contract frame.
"""

# Undo the anisotropic pixel normalisation in analyze.py (x/608, y/1080).
LEGACY_XZ_SCALE = 608 / 1080


def parse(text_or_lines):
    """Parse coordinate lines into a list of (x, y, z) tuples or None."""
    if isinstance(text_or_lines, str):
        text_or_lines = text_or_lines.split("\n")
    out = []
    for line in text_or_lines:
        line = line.strip()
        if not line:
            continue
        if line == "null":
            out.append(None)
            continue
        p = line.split(",")
        out.append((float(p[0]), float(p[1]), float(p[2])))
    return out


def adapt_legacy(coords, xz_scale=LEGACY_XZ_SCALE):
    """Map legacy coords_compact.txt positions into the contract frame."""
    pts = [c for c in coords if c is not None]
    if not pts:
        return coords
    ys = [c[1] for c in pts]
    y0 = min(ys)
    height = (max(ys) - y0) or 1.0
    # Trunk axis: centroid of the horizontal positions.
    cx = sum(c[0] for c in pts) / len(pts)
    cz = sum(c[2] for c in pts) / len(pts)
    k = xz_scale / height
    out = []
    for c in coords:
        if c is None:
            out.append(None)
        else:
            out.append(((c[0] - cx) * k, (c[1] - y0) / height, (c[2] - cz) * k))
    return out


def pad(coords, n):
    """Truncate or pad with None to exactly n entries."""
    coords = list(coords[:n])
    while len(coords) < n:
        coords.append(None)
    return coords


def load(path, n, legacy=None):
    """Load coordinates from path for n LEDs; returns None if unreadable.
    legacy defaults to True for files named coords_compact.txt."""
    try:
        with open(path) as f:
            coords = parse(f.read())
    except OSError:
        return None
    if legacy is None:
        legacy = path.endswith("coords_compact.txt")
    if legacy:
        coords = adapt_legacy(coords)
    return pad(coords, n)
