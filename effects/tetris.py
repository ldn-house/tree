"""Tetris on the cone: tetrominoes fall down the tree, stack and clear.

The tree's surface is a coarse cylindrical grid, ROWS height bands by
COLS sectors that wrap round the trunk. Rows hold equal numbers of LEDs
(the tree is denser low down); sector edges are placed halfway between
equal angles and the edges that would split each row's LEDs evenly, so
cells hold a few LEDs each without the columns wandering far (4-12 LEDs
per cell on the current calibration).

A deterministic auto-player picks each piece's rotation and column by the
usual heuristic (lines cleared, aggregate height, holes, bumpiness) plus
seeded noise set by `skill`, so the stack rises and falls and now and
then tops out. One rotation is scored per frame to keep frames short. The
piece drops one row per beat and locks, completed rows flash white once
and clear, and when a piece can't enter at the top the stack fades out
and a new game starts. Pieces come from a shuffled
7-bag drawn with ctx.random.
"""

from effects._util import ONE, MASK, Layout, Param

NAME = "tetris"
DURATION_MS = 40000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
    "skill": (0.75, 0.0, 1.0),  # 1 = always the best placement, 0 = noisy
}
NEEDS_COORDS = True

ROWS = 10
COLS = 6
CELLS = ROWS * COLS

DROP_MS = 300  # one row per beat at speed 1
LOCK_MS = 160  # a landed piece stays bright this long before the next one
FLASH_MS = 650  # completed rows: flash white once, then fade out
OVER_MS = 1600  # top out: the stack fades away
REST_MS = 500  # dark pause before the next game
LANDED = 150  # /256 brightness of the stack; the falling piece is 256
EMPTY = (0, 0, 2)  # empty cells, so the tree's outline stays visible

# Tetromino colours, the usual ones: I O T S Z J L.
COLOURS = bytes((
    0, 210, 255,
    255, 190, 0,
    160, 0, 255,
    0, 255, 30,
    255, 0, 0,
    0, 50, 255,
    255, 90, 0,
))

# Base shapes as (x, y) cells, y up.
SHAPES = (
    ((0, 0), (1, 0), (2, 0), (3, 0)),  # I
    ((0, 0), (1, 0), (0, 1), (1, 1)),  # O
    ((0, 0), (1, 0), (2, 0), (1, 1)),  # T
    ((0, 0), (1, 0), (1, 1), (2, 1)),  # S
    ((1, 0), (2, 0), (0, 1), (1, 1)),  # Z
    ((0, 0), (1, 0), (2, 0), (0, 1)),  # J
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
)

# Placement heuristic weights (Dellacherie / El-Tetris style, x1000).
W_LINES = 760
W_HEIGHT = -510
W_HOLES = -3560
W_BUMP = -184
NOISE = 8000  # score noise at skill 0 (one cleared line is 760)

_FALL = 0
_LOCK = 1
_FLASH = 2
_OVER = 3
_REST = 4
_THINK = 5


def _rotations():
    """(nrot, offs, ph): distinct rotations per piece, their cells as
    x, y byte pairs (normalised to min 0) and their heights."""
    nrot = bytearray(7)
    offs = bytearray(7 * 4 * 8)
    ph = bytearray(7 * 4)
    for p in range(7):
        cells = SHAPES[p]
        seen = []
        for _ in range(4):
            mx = min(c[0] for c in cells)
            my = min(c[1] for c in cells)
            norm = sorted((c[0] - mx, c[1] - my) for c in cells)
            if norm not in seen:
                seen.append(norm)
            cells = [(c[1], -c[0]) for c in cells]
        nrot[p] = len(seen)
        for r in range(len(seen)):
            o = (p * 4 + r) * 8
            for c in range(4):
                offs[o + 2 * c] = seen[r][c][0]
                offs[o + 2 * c + 1] = seen[r][c][1]
            ph[p * 4 + r] = max(c[1] for c in seen[r]) + 1
    return nrot, offs, ph


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
        lay = Layout(ctx)
        self.known = lay.known
        self.cell = bytearray(n)
        self.map_cells(lay)
        self.nrot, self.offs, self.ph = _rotations()
        self.board = bytearray(CELLS)
        self.colour = bytearray(3 * CELLS)
        self.heights = bytearray(COLS)
        self.full = bytearray(ROWS)
        self.bag = bytearray(7)
        self.bag_i = 7
        self.speed = Param(ctx, "speed", 1.0)
        self.skill = Param(ctx, "skill", 0.75, 256)
        self.skill_v = self.skill.get()
        self.last = 0
        self.sub = 0
        self.t = 0  # ms in the current state, at speed 1
        self.state = _REST
        self.p = 0
        self.r = 0
        self.x = 0
        self.y = 0
        self.eval_r = 0
        self.best = 0

    def map_cells(self, lay):
        n = self.n
        cell = self.cell
        idx = [i for i in range(n) if lay.known[i]]
        for i in range(n):
            cell[i] = 255
        if not idx:
            return
        # Rows: equal LED counts by height.
        idx.sort(key=lambda i: lay.h[i])
        m = len(idx)
        rows = [[] for _ in range(ROWS)]
        for j in range(m):
            rows[j * ROWS // m].append(idx[j])
        # Sectors: halfway between equal angles and the row's quantiles.
        for row in range(ROWS):
            members = rows[row]
            members.sort(key=lambda i: lay.theta[i])
            k = len(members)
            if not k:
                continue
            bounds = [0] * (COLS + 1)
            for s in range(1, COLS):
                q = lay.theta[members[s * k // COLS]]
                bounds[s] = (s * ONE // COLS + q) >> 1
            bounds[COLS] = ONE
            for i in members:
                s = 0
                while s < COLS - 1 and lay.theta[i] >= bounds[s + 1]:
                    s += 1
                cell[i] = row * COLS + s

    # ─── Game logic (integer only, no allocation) ─────────────────────────

    def fits(self, p, r, x, y):
        offs = self.offs
        board = self.board
        o = (p * 4 + r) * 8
        for c in range(4):
            cy = y + offs[o + 2 * c + 1]
            if cy < 0:
                return False
            if cy < ROWS and board[cy * COLS + (x + offs[o + 2 * c]) % COLS]:
                return False
        return True

    def stamp(self, p, r, x, y, v):
        offs = self.offs
        board = self.board
        o = (p * 4 + r) * 8
        for c in range(4):
            cy = y + offs[o + 2 * c + 1]
            if cy < ROWS:
                board[cy * COLS + (x + offs[o + 2 * c]) % COLS] = v

    def drop_y(self, p, r, x):
        """Resting y for a piece dropped straight from the top, or -1."""
        y = ROWS - self.ph[p * 4 + r]
        if not self.fits(p, r, x, y):
            return -1
        while self.fits(p, r, x, y - 1):
            y -= 1
        return y

    def score(self):
        board = self.board
        hts = self.heights
        lines = 0
        for row in range(ROWS):
            b = row * COLS
            f = 1
            for c in range(COLS):
                if not board[b + c]:
                    f = 0
                    break
            lines += f
        agg = 0
        holes = 0
        for c in range(COLS):
            hgt = 0
            for row in range(ROWS - 1, -1, -1):
                if board[row * COLS + c]:
                    hgt = row + 1
                    break
            hts[c] = hgt
            agg += hgt
            for row in range(hgt):
                if not board[row * COLS + c]:
                    holes += 1
        bump = 0
        for c in range(COLS):
            d = hts[c] - hts[(c + 1) % COLS]
            bump += d if d >= 0 else -d
        agg -= lines * COLS
        return W_LINES * lines + W_HEIGHT * agg + W_HOLES * holes + W_BUMP * bump

    def next_piece(self):
        bag = self.bag
        if self.bag_i >= 7:
            for i in range(7):
                bag[i] = i
            for i in range(6, 0, -1):
                j = self.rand(8) % (i + 1)
                t = bag[i]
                bag[i] = bag[j]
                bag[j] = t
            self.bag_i = 0
        p = bag[self.bag_i]
        self.bag_i += 1
        return p

    def think(self):
        """Score every column for the next rotation of the waiting piece;
        one rotation per frame keeps each frame short on the Pico. Returns
        True once every rotation has been tried."""
        p = self.p
        r = self.eval_r
        noise = (256 - self.skill_v) * NOISE >> 8
        bits = 0
        while (1 << bits) <= noise:
            bits += 1
        for x in range(COLS):
            y = self.drop_y(p, r, x)
            if y < 0:
                continue
            self.stamp(p, r, x, y, 1)
            s = self.score()
            self.stamp(p, r, x, y, 0)
            if noise:
                s += self.rand(bits) * noise >> bits
            if s > self.best:
                self.best = s
                self.r = r
                self.x = x
        self.eval_r = r + 1
        return self.eval_r >= self.nrot[p]

    def clear_rows(self):
        """Mark full rows in self.full; return how many."""
        board = self.board
        count = 0
        for row in range(ROWS):
            b = row * COLS
            f = 1
            for c in range(COLS):
                if not board[b + c]:
                    f = 0
                    break
            self.full[row] = f
            count += f
        return count

    def collapse(self):
        board = self.board
        dst = 0
        for row in range(ROWS):
            if self.full[row]:
                continue
            if dst != row:
                for c in range(COLS):
                    board[dst * COLS + c] = board[row * COLS + c]
            dst += 1
        for row in range(dst, ROWS):
            self.full[row] = 0
            for c in range(COLS):
                board[row * COLS + c] = 0
        for row in range(ROWS):
            self.full[row] = 0

    def step(self, dt):
        """Advance the game by dt ms (speed 1)."""
        self.t += dt
        st = self.state
        if st == _FALL:
            while self.t >= DROP_MS and self.state == _FALL:
                self.t -= DROP_MS
                if self.fits(self.p, self.r, self.x, self.y - 1):
                    self.y -= 1
                else:
                    self.stamp(self.p, self.r, self.x, self.y, self.p + 1)
                    self.state = _LOCK
                    self.t = 0
        elif st == _LOCK:
            if self.t >= LOCK_MS:
                self.t = 0
                if self.clear_rows():
                    self.state = _FLASH
                else:
                    self.begin_piece()
        elif st == _FLASH:
            if self.t >= FLASH_MS:
                self.collapse()
                self.t = 0
                self.begin_piece()
        elif st == _THINK:
            if self.think():
                self.t = 0
                if self.r < 0:
                    self.state = _OVER  # no column fits: top out
                else:
                    self.y = ROWS - self.ph[self.p * 4 + self.r]
                    self.state = _FALL
        elif st == _OVER:
            if self.t >= OVER_MS:
                board = self.board
                for i in range(CELLS):
                    board[i] = 0
                self.state = _REST
                self.t = 0
        elif self.t >= REST_MS:
            self.t = 0
            self.begin_piece()

    def begin_piece(self):
        self.p = self.next_piece()
        self.eval_r = 0
        self.best = -(1 << 29)
        self.r = -1
        self.state = _THINK
        self.t = 0

    # ─── Rendering ───────────────────────────────────────────────────────

    def render(self, t_ms, frame):
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.sub += dt * self.speed.get()
        # Read here, every frame, so a changed value converts in render()
        # (Param caches it) rather than in the middle of think().
        self.skill_v = self.skill.get()
        self.step(self.sub >> 12)
        self.sub &= MASK

        st = self.state
        t = self.t
        # Stack brightness: dims to nothing over a top out.
        lv = LANDED
        if st == _OVER:
            q = (OVER_MS - t) * 256 // OVER_MS
            lv = LANDED * (q * q >> 8) >> 8
        # Completed rows: white attack, then fade out.
        fw = 0
        fl = 0
        if st == _FLASH:
            attack = FLASH_MS >> 2
            if t < attack:
                fw = 256 * t // attack
                fl = 256
            else:
                q = (FLASH_MS - t) * 256 // (FLASH_MS - attack)
                fw = q
                fl = q * q >> 8
        er, eg, eb = EMPTY
        board = self.board
        col = self.colour
        full = self.full
        for row in range(ROWS):
            flashing = full[row] and st == _FLASH
            for c in range(COLS):
                i = row * COLS + c
                j = 3 * i
                v = board[i]
                if not v:
                    col[j] = er
                    col[j + 1] = eg
                    col[j + 2] = eb
                    continue
                o = 3 * (v - 1)
                r = COLOURS[o]
                g = COLOURS[o + 1]
                b = COLOURS[o + 2]
                if flashing:
                    r = (r + ((255 - r) * fw >> 8)) * fl >> 8
                    g = (g + ((255 - g) * fw >> 8)) * fl >> 8
                    b = (b + ((255 - b) * fw >> 8)) * fl >> 8
                else:
                    r = r * lv >> 8
                    g = g * lv >> 8
                    b = b * lv >> 8
                col[j] = r
                col[j + 1] = g
                col[j + 2] = b
        if st == _FALL or st == _LOCK:
            # The live piece at full colour; just landed, it's still the
            # brightest thing on the tree until LOCK_MS passes.
            o = 3 * self.p
            r = COLOURS[o]
            g = COLOURS[o + 1]
            b = COLOURS[o + 2]
            offs = self.offs
            po = (self.p * 4 + self.r) * 8
            for c in range(4):
                cy = self.y + offs[po + 2 * c + 1]
                if cy < ROWS:
                    j = 3 * (cy * COLS + (self.x + offs[po + 2 * c]) % COLS)
                    col[j] = r
                    col[j + 1] = g
                    col[j + 2] = b

        cell = self.cell
        for i in range(self.n):
            k = 3 * i
            c = cell[i]
            if c == 255:
                frame[k] = 0
                frame[k + 1] = 0
                frame[k + 2] = 0
            else:
                j = 3 * c
                frame[k] = col[j]
                frame[k + 1] = col[j + 1]
                frame[k + 2] = col[j + 2]
