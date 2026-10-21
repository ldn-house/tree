# DISABLED: not in effects.EFFECTS. To re-enable, move "space_invaders" from DISABLED back into EFFECTS in effects/__init__.py.
"""Space Invaders on the cone: a block of invaders marches round the tree.

The tree's surface is a coarse cylindrical grid, ROWS height bands by COLS
sectors that wrap round the trunk (the same mapping as tetris: rows hold
equal numbers of LEDs, and sector edges sit halfway between equal angles
and the row's LED quantiles, giving 1-10 LEDs per cell on the current
calibration).

A block of green invaders, one every other column, marches sideways
(round the trunk), steps down a row at each end of its sweep, and speeds
up as its numbers fall. A red cannon on the bottom row picks a target,
leads it, and fires white bolts up the tree; hits burst once in orange.
Now and then an invader drops a yellow bomb, which knocks the cannon out
for a moment if it lands. A cleared wave starts a new, slightly faster
one; a wave that reaches the bottom flashes red once and starts again.
"""

from array import array

from effects._util import ONE, MASK, Layout, Param

NAME = "space_invaders"
DURATION_MS = 40000
PARAMS = {
    "speed": (1.0, 0.25, 3.0),
}
NEEDS_COORDS = True

ROWS = 10
COLS = 10
CELLS = ROWS * COLS

FORM_ROWS = 4
FORM_COLS = 3  # invaders per row, every other column
TOTAL = FORM_ROWS * FORM_COLS
SWEEP = 5  # columns marched each way before stepping down

STEP_SLOW = 650  # march step with the whole block alive, ms
STEP_FAST = 150  # with one left
WAVE_FASTER = 30  # /256 faster per wave, up to MAX_WAVE waves
MAX_WAVE = 5
BOLT_MS = 65  # a bolt climbs one row in this long
COOLDOWN_MS = 600  # between shots
CANNON_MS = 110  # the cannon moves one column in this long
BOMB_MS = 150  # a bomb falls one row in this long
BOMB_GAP = 1800  # mean ms between bombs
DEAD_MS = 1300  # the cannon is out this long after a bomb hit
BURST_MS = 450  # a hit's burst
SLOTS = 4  # bursts alive at once
CLEARED_MS = 1200  # pause after a cleared wave
LANDED_MS = 1600  # red flash and pause after the invaders land

INVADER = (40, 255, 50)
POSE = 175  # /256: invaders alternate between full and this each step
CANNON = (255, 30, 0)
BOLT = (255, 255, 255)
BOMB = (255, 170, 0)
BURST = (255, 150, 40)
DEATH = (255, 20, 0)
EMPTY = (0, 0, 2)

_PLAY = 0
_CLEARED = 1
_LANDED = 2


class Effect:
    def __init__(self, ctx):
        n = ctx.n
        self.n = n
        self.rand = ctx.random.getrandbits
        lay = Layout(ctx)
        self.cell = bytearray(n)
        self.map_cells(lay)
        self.col = bytearray(3 * CELLS)
        self.alive = bytearray(TOTAL)
        self.burst_cell = bytearray(SLOTS)
        self.burst_t = array("i", bytes(4 * SLOTS))
        self.burst_red = bytearray(SLOTS)
        self.speed = Param(ctx, "speed", 1.0)
        self.last = 0
        self.sub = 0
        self.t = 0  # game clock, ms at speed 1
        self.wave = 0
        self.state_t = 0
        self.new_wave()
        self.cannon = 0
        self.cannon_mv = 0
        self.dead_until = 0
        self.bolt_row = -1
        self.bolt_col = 0
        self.bolt_t = 0
        self.fired = -COOLDOWN_MS
        self.bomb_row = -1
        self.bomb_col = 0
        self.bomb_t = 0
        self.next_bomb = BOMB_GAP

    def map_cells(self, lay):
        n = self.n
        cell = self.cell
        idx = [i for i in range(n) if lay.known[i]]
        for i in range(n):
            cell[i] = 255
        if not idx:
            return
        idx.sort(key=lambda i: lay.h[i])
        m = len(idx)
        rows = [[] for _ in range(ROWS)]
        for j in range(m):
            rows[j * ROWS // m].append(idx[j])
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

    def new_wave(self):
        for j in range(TOTAL):
            self.alive[j] = 1
        self.count = TOTAL
        self.fx = self.rand(4) % COLS  # block's left column
        self.fy = ROWS - 1  # block's top row
        self.dir = 1
        self.swept = 0
        self.poses = 0
        self.step_t = self.t
        self.state = _PLAY
        self.bolt_row = -1
        self.bomb_row = -1

    def step_ms(self):
        s = STEP_FAST + (STEP_SLOW - STEP_FAST) * (self.count - 1) // (TOTAL - 1)
        w = self.wave if self.wave < MAX_WAVE else MAX_WAVE
        return s - (s * WAVE_FASTER * w >> 8)

    def inv_col(self, j):
        return (self.fx + 2 * (j % FORM_COLS)) % COLS

    def inv_row(self, j):
        return self.fy - j // FORM_COLS

    def invader_at(self, row, c):
        """Index of the live invader in that cell, or -1."""
        r = self.fy - row
        if r < 0 or r >= FORM_ROWS:
            return -1
        d = (c - self.fx) % COLS
        if d & 1 or d >= 2 * FORM_COLS:
            return -1
        j = r * FORM_COLS + (d >> 1)
        return j if self.alive[j] else -1

    def burst(self, c, red):
        # Reuse the oldest slot.
        s = 0
        for k in range(1, SLOTS):
            if self.burst_t[k] < self.burst_t[s]:
                s = k
        self.burst_cell[s] = c
        self.burst_t[s] = self.t
        self.burst_red[s] = red

    def march(self):
        if self.swept >= SWEEP:
            self.fy -= 1
            self.swept = 0
            self.dir = -self.dir
        else:
            self.fx = (self.fx + self.dir) % COLS
            self.swept += 1
        self.poses ^= 1
        # Landed: the lowest live invader reaches the cannon's row.
        for j in range(TOTAL):
            if self.alive[j] and self.inv_row(j) <= 0:
                self.state = _LANDED
                self.state_t = self.t
                self.bolt_row = -1
                self.bomb_row = -1
                return

    def aim(self):
        """Pick the live invader the cannon can reach soonest, leading its
        column by how far the block will march while the bolt climbs."""
        best = -1
        bd = 99
        step = self.step_ms()
        for j in range(TOTAL):
            if not self.alive[j]:
                continue
            row = self.inv_row(j)
            # Only shoot at the lowest invader in each column.
            if row > 0 and self.invader_at(row - 1, self.inv_col(j)) >= 0:
                continue
            lead = row * BOLT_MS // step
            left = SWEEP - self.swept
            c = self.inv_col(j) + self.dir * (lead if lead < left else left)
            d = (c - self.cannon) % COLS
            if d > COLS // 2:
                d = COLS - d
            if d < bd:
                bd = d
                best = c % COLS
        return best

    def step(self, dt):
        self.t += dt
        t = self.t
        if self.state != _PLAY:
            wait = CLEARED_MS if self.state == _CLEARED else LANDED_MS
            if t - self.state_t >= wait:
                if self.state == _CLEARED:
                    self.wave += 1
                else:
                    self.wave = 0
                self.new_wave()
            return
        # March.
        step = self.step_ms()
        if t - self.step_t >= step:
            self.step_t = t
            self.march()
            if self.state != _PLAY:
                return
        # Cannon: move towards its target and fire when lined up.
        if t >= self.dead_until:
            target = self.aim()
            if target >= 0 and t - self.cannon_mv >= CANNON_MS:
                d = (target - self.cannon) % COLS
                if d:
                    self.cannon = (self.cannon + (1 if d <= COLS // 2 else -1)) % COLS
                    self.cannon_mv = t
            if (target == self.cannon and self.bolt_row < 0
                    and t - self.fired >= COOLDOWN_MS):
                self.bolt_row = 1
                self.bolt_col = self.cannon
                self.bolt_t = t
                self.fired = t
        # Bolt.
        while self.bolt_row >= 0 and t - self.bolt_t >= BOLT_MS:
            self.bolt_t += BOLT_MS
            j = self.invader_at(self.bolt_row, self.bolt_col)
            if j < 0 and self.bolt_row + 1 < ROWS:
                self.bolt_row += 1
                j = self.invader_at(self.bolt_row, self.bolt_col)
            elif j < 0:
                self.bolt_row = -1
                break
            if j >= 0:
                self.alive[j] = 0
                self.count -= 1
                self.burst(self.bolt_row * COLS + self.bolt_col, 0)
                self.bolt_row = -1
                if not self.count:
                    self.state = _CLEARED
                    self.state_t = t
                    self.bomb_row = -1
                    return
        # Bombs: from the lowest invader in a random live column.
        if self.bomb_row < 0 and t >= self.next_bomb:
            j = self.rand(5) % TOTAL
            k = 0
            while k < TOTAL and not self.alive[j]:
                j = (j + 1) % TOTAL
                k += 1
            c = self.inv_col(j)
            row = self.inv_row(j)
            while row > 0 and self.invader_at(row - 1, c) >= 0:
                row -= 1
            self.bomb_row = row - 1
            self.bomb_col = c
            self.bomb_t = t
            self.next_bomb = t + BOMB_GAP // 2 + self.rand(10) * BOMB_GAP // 1024
        if self.bomb_row >= 0 and t - self.bomb_t >= BOMB_MS:
            self.bomb_t = t
            self.bomb_row -= 1
            if self.bomb_row == 0 and self.bomb_col == self.cannon and t >= self.dead_until:
                self.burst(self.cannon, 1)
                self.dead_until = t + DEAD_MS
                self.bomb_row = -1
            elif self.bomb_row < 0:
                self.bomb_row = -1

    # ─── Rendering ───────────────────────────────────────────────────────

    def put(self, c, r, g, b):
        j = 3 * c
        col = self.col
        col[j] = r
        col[j + 1] = g
        col[j + 2] = b

    def add(self, c, r, g, b):
        j = 3 * c
        col = self.col
        v = col[j] + r
        col[j] = v if v < 255 else 255
        v = col[j + 1] + g
        col[j + 1] = v if v < 255 else 255
        v = col[j + 2] + b
        col[j + 2] = v if v < 255 else 255

    def render(self, t_ms, frame):
        dt = t_ms - self.last
        self.last = t_ms
        if dt < 0 or dt > 1000:
            dt = 0 if dt < 0 else 1000
        self.sub += dt * self.speed.get()
        self.step(self.sub >> 12)
        self.sub &= MASK
        t = self.t

        er, eg, eb = EMPTY
        for c in range(CELLS):
            self.put(c, er, eg, eb)
        # Invaders, in two poses; on landing they flash red once and fade.
        ir, ig, ib = INVADER
        lv = 256 if self.poses else POSE
        if self.state == _LANDED:
            e = t - self.state_t
            q = (LANDED_MS - e) * 256 // LANDED_MS if e < LANDED_MS else 0
            ir, ig, ib = DEATH
            lv = q * q >> 8
        if self.state != _CLEARED:
            for j in range(TOTAL):
                if self.alive[j]:
                    row = self.inv_row(j)
                    if 0 <= row < ROWS:
                        self.put(row * COLS + self.inv_col(j), ir * lv >> 8, ig * lv >> 8, ib * lv >> 8)
        # Cannon (dark while knocked out), bolt and bomb.
        if t >= self.dead_until and self.state != _LANDED:
            cr, cg, cb = CANNON
            self.put(self.cannon, cr, cg, cb)
            # Dimmer shoulders either side, so it reads as a base.
            cr = cr * 90 >> 8
            cg = cg * 90 >> 8
            cb = cb * 90 >> 8
            self.put((self.cannon + 1) % COLS, cr, cg, cb)
            self.put((self.cannon - 1) % COLS, cr, cg, cb)
        if self.bolt_row >= 0:
            r, g, b = BOLT
            self.put(self.bolt_row * COLS + self.bolt_col, r, g, b)
        if self.bomb_row >= 0:
            r, g, b = BOMB
            self.put(self.bomb_row * COLS + self.bomb_col, r, g, b)
        # Bursts: the cell and its four neighbours, fading once.
        for s in range(SLOTS):
            e = t - self.burst_t[s]
            if e < 0 or e >= BURST_MS:
                continue
            q = (BURST_MS - e) * 256 // BURST_MS
            q = q * q >> 8
            if self.burst_red[s]:
                r, g, b = DEATH
            else:
                r, g, b = BURST
            r = r * q >> 8
            g = g * q >> 8
            b = b * q >> 8
            c = self.burst_cell[s]
            row = c // COLS
            k = c % COLS
            self.add(c, r, g, b)
            r >>= 1
            g >>= 1
            b >>= 1
            self.add(row * COLS + (k + 1) % COLS, r, g, b)
            self.add(row * COLS + (k - 1) % COLS, r, g, b)
            if row + 1 < ROWS:
                self.add(c + COLS, r, g, b)
            if row > 0:
                self.add(c - COLS, r, g, b)

        cell = self.cell
        col = self.col
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
