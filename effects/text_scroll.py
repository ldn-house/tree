"""Text scrolling round the tree, readable from one side.

The text occupies an arc of the tree (120 degrees by default) facing the
viewer. Each LED samples the 5x5 font bitmap bilinearly at its projected
position rather than snapping to a cell, so sparse areas still read as
letters (review M10). facing turns the text round the tree (0 = towards the
front camera); mirror flips the reading direction in case the calibration's
handedness is reversed.
"""

from array import array

from effects._util import ONE, MASK, Layout, Param

NAME = "text_scroll"
DURATION_MS = 8000
PARAMS = {
    "speed": (1.0, 0.1, 5.0),
    "facing": (0, -180, 180),  # degrees round the tree
    "mirror": (0, 0, 1),
    "arc": (120, 60, 360),  # degrees of the tree used as the display
    "columns": (8, 4, 24),  # font columns visible across the arc
}
NEEDS_COORDS = True

TEXT = "HI"
COLOR = (255, 0, 0)
COLUMN_MS = 90  # time to scroll one font column at speed 1
BOTTOM = 0.2  # text band, as fractions of the tree height
TOP = 0.8
NONE = -32768

# Wide 5x5 block font, one byte per column, bit 0 = top row. Blank columns
# inside letters keep strokes apart on the sparse LED grid.
FONT = {
    "A": (0x1E, 0x00, 0x05, 0x00, 0x1E),
    "B": (0x1F, 0x00, 0x15, 0x00, 0x0A),
    "C": (0x0E, 0x00, 0x11, 0x00, 0x11),
    "D": (0x1F, 0x00, 0x11, 0x00, 0x0E),
    "E": (0x1F, 0x00, 0x15, 0x00, 0x11),
    "F": (0x1F, 0x00, 0x05, 0x00, 0x01),
    "G": (0x0E, 0x00, 0x11, 0x00, 0x1D),
    "H": (0x1F, 0x00, 0x04, 0x00, 0x1F),
    "I": (0x11, 0x00, 0x1F, 0x00, 0x11),
    "J": (0x08, 0x00, 0x10, 0x00, 0x0F),
    "K": (0x1F, 0x00, 0x04, 0x00, 0x1B),
    "L": (0x1F, 0x00, 0x10, 0x00, 0x10),
    "M": (0x1F, 0x02, 0x00, 0x02, 0x1F),
    "N": (0x1F, 0x02, 0x04, 0x08, 0x1F),
    "O": (0x0E, 0x00, 0x11, 0x00, 0x0E),
    "P": (0x1F, 0x00, 0x05, 0x00, 0x02),
    "Q": (0x0E, 0x00, 0x19, 0x00, 0x1E),
    "R": (0x1F, 0x00, 0x05, 0x00, 0x1A),
    "S": (0x12, 0x00, 0x15, 0x00, 0x09),
    "T": (0x01, 0x00, 0x1F, 0x00, 0x01),
    "U": (0x0F, 0x00, 0x10, 0x00, 0x0F),
    "V": (0x07, 0x00, 0x18, 0x00, 0x07),
    "W": (0x1F, 0x08, 0x00, 0x08, 0x1F),
    "X": (0x1B, 0x00, 0x04, 0x00, 0x1B),
    "Y": (0x03, 0x00, 0x1C, 0x00, 0x03),
    "Z": (0x19, 0x00, 0x15, 0x00, 0x13),
    " ": (0x00, 0x00, 0x00, 0x00, 0x00),
    "!": (0x00, 0x00, 0x17, 0x00, 0x00),
}
GAP = 3  # blank columns after each character


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        text = ctx.params.get("text")
        if not isinstance(text, str):
            text = TEXT
        cols = []
        for ch in text.upper():
            glyph = FONT.get(ch)
            if glyph:
                cols.extend(glyph)
                cols.extend((0,) * GAP)
        # Shifted up one bit so rows -1 and 5 read as blank in render().
        self.cols = bytearray([c << 1 for c in cols])
        lay = Layout(ctx)
        self.known = lay.known
        self.theta = lay.theta
        # Font row position of each LED in 1/256 rows, NONE outside the band.
        self.v = array("h", bytes(2 * n))
        bottom = int(BOTTOM * ONE)
        top = int(TOP * ONE)
        for i in range(n):
            h = lay.h[i]
            if lay.known[i] and bottom <= h <= top:
                self.v[i] = (top - h) * 5 * 256 // (top - bottom)
            else:
                self.v[i] = NONE
        # Font column position across the display arc, in 1/256 columns.
        self.u = array("h", bytes(2 * n))
        self.speed = Param(ctx, "speed", 1.0)
        self.facing = Param(ctx, "facing", 0, 1)
        self.mirror = Param(ctx, "mirror", 0, 1)
        self.arc = Param(ctx, "arc", 120, 1)
        self.columns = Param(ctx, "columns", 8, 1)
        self.key = None
        self.scroll = 0
        self.last = 0

    def _project(self, facing, mirror, arc, columns):
        # Viewer at azimuth `facing` looks at the trunk; their right-hand
        # side is decreasing theta (theta = atan2(z, x)).
        face = (270 - facing) * ONE // 360 & MASK
        span = max(1, arc * ONE // 360)
        half = ONE >> 1
        theta = self.theta
        known = self.known
        u = self.u
        for i in range(self.n):
            d = ((theta[i] - face + half) & MASK) - half
            if not mirror:
                d = -d
            d += span >> 1
            if known[i] and 0 <= d < span:
                u[i] = d * columns * 256 // span
            else:
                u[i] = NONE

    def render(self, t_ms, frame):
        facing = self.facing.get()
        mirror = self.mirror.get()
        arc = self.arc.get()
        columns = self.columns.get()
        key = ((facing * 2 + mirror) * 400 + arc) * 32 + columns
        if key != self.key:
            self.key = key
            self._project(facing, mirror, arc, columns)
        cols = self.cols
        width = len(cols)
        total = (width + columns) * ONE
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0
        # scroll position in 1/ONE columns
        self.scroll = (self.scroll + dt * self.speed.get() // COLUMN_MS) % total
        start = (self.scroll >> 4) - columns * 256 - 128
        cr, cg, cb = COLOR
        u = self.u
        v = self.v
        for i in range(self.n):
            k = 3 * i
            ui = u[i]
            vi = v[i]
            if ui == NONE or vi == NONE:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
                continue
            cx = start + ui
            c0 = cx >> 8
            fx = cx & 255
            ry = vi - 128
            r0 = (ry >> 8) + 1
            fy = ry & 255
            a = cols[c0] if 0 <= c0 < width else 0
            b = cols[c0 + 1] if 0 <= c0 + 1 < width else 0
            top = ((a >> r0) & 1) * (256 - fx) + ((b >> r0) & 1) * fx
            bot = ((a >> (r0 + 1)) & 1) * (256 - fx) + ((b >> (r0 + 1)) & 1) * fx
            s = (top * (256 - fy) + bot * fy) >> 8
            if s > 255:
                s = 255
            frame[k] = cr * s >> 8
            frame[k + 1] = cg * s >> 8
            frame[k + 2] = cb * s >> 8
