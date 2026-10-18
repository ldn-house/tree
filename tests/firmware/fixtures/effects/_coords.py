"""Minimal stand-in for effects/_coords.py (same API)."""


def parse(text_or_lines):
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


def adapt_legacy(coords):
    pts = [c for c in coords if c is not None]
    if not pts:
        return coords
    ys = [c[1] for c in pts]
    y0 = min(ys)
    h = (max(ys) - y0) or 1.0
    return [None if c is None else (c[0] / h, (c[1] - y0) / h, c[2] / h) for c in coords]


def pad(coords, n):
    coords = list(coords[:n])
    while len(coords) < n:
        coords.append(None)
    return coords


def load(path, n, legacy=None):
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
