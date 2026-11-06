"""Every displayed frame, on MicroPython, with default colour accuracy.

Drive the real engine loop with a virtual clock so no 40 fps frame is missed.
"""
import sys
import os
import time
import asyncio

repo = sys.argv[1]
sys.path.insert(0, repo + '/firmware')
sys.path.insert(0, repo + '/tests/firmware/mocks')
sys.path.insert(0, repo)
os.chdir(sys.argv[2])
from app import engine, transitions
from app.engine import Engine
from app.state import State
from effects._output import Output, ColourFloor, COLOUR_ACCURACY, DITHER_MIN, drift

WARM = (255, 87, 14)


class Warm:
    def __init__(self, ctx):
        pass

    def render(self, t, raw):
        for k in range(0, len(raw), 3):
            raw[k] = 255
            raw[k + 1] = 87
            raw[k + 2] = 14


class Cool:
    def __init__(self, ctx):
        pass

    def render(self, t, raw):
        for k in range(0, len(raw), 3):
            raw[k] = 0
            raw[k + 1] = 0
            raw[k + 2] = 255


class Module:
    Effect = Warm


class CoolModule:
    Effect = Cool


class FX:
    def names(self):
        return ['warm', 'other', 'third']

    def load(self, name):
        return CoolModule if name == 'other' else Module

    def unload(self, name):
        pass

    def meta(self, name):
        return {'params': {}, 'duration_ms': 10000, 'needs_coords': False}


class Driver:
    def __init__(self):
        self.frames = []

    def busy(self):
        return False

    def write(self, wire):
        self.frames.append((clock, bytes(wire)))


class Finished(Exception):
    pass


clock = 0
class Clock:
    ticks_ms = staticmethod(lambda: clock)
    ticks_us = staticmethod(time.ticks_us)
    ticks_diff = staticmethod(time.ticks_diff)


engine.time = Clock
transitions.time = Clock
# The colour analysis is paced by real time, which a busy machine stretches over
# more frames. Finish it within one frame, so these frames depend only on the clock.
engine.COLOUR_US = 0


def run(commands, end):
    global clock
    clock = 0
    s = State()
    s.update({'effect': 'warm', 'power': True, 'brightness': 255})
    out = Output(500)
    driver = Driver()
    e = Engine(500, s, FX(), out, driver, 40)
    e.colour = ColourFloor(out)
    Warm(None).render(0, e.raw)
    e.colour.add(e.raw)
    assert e.colour.accuracy == COLOUR_ACCURACY
    assert 15 < out.floor_pct() < 17
    commands = list(commands)

    async def advance(ms):
        global clock
        clock += ms
        if clock > end:
            raise Finished()
        while commands and clock >= commands[0][0]:
            at, values = commands.pop(0)
            s.update(values)

    asyncio.sleep_ms = advance
    try:
        asyncio.run(e.run())
    except Finished:
        pass
    assert not commands
    return driver.frames


def accurate(frames):
    maximum = 0
    worst = None
    for at, wire in frames:
        assert sum(wire) <= 47812
        for k in range(0, len(wire), 3):
            rgb = tuple(wire[k:k + 3])
            if not any(rgb):
                continue
            delta = drift(rgb, WARM)
            if delta > maximum:
                maximum = delta
                worst = (at, rgb)
        # ColourFloor models channels above DITHER_MIN by their averaged
        # spatial-dither value. Check that contract from the wire too.
        mean = tuple(sum(wire[k] for k in range(c, len(wire), 3)) / 500 for c in range(3))
        assert drift(mean, WARM) <= COLOUR_ACCURACY, (at, mean, drift(mean, WARM))
        # For this solid colour, every channel below DITHER_MIN must be
        # uniform across the tree, even while the red channel dithers.
        # At the full power cap the existing limiter may trim individual
        # channels; check the low-end rule away from that cap.
        if sum(wire) == 47812:
            continue
        for channel in range(3):
            values = [wire[k] for k in range(channel, len(wire), 3)]
            if max(values) < DITHER_MIN:
                assert min(values) == max(values), (at, channel)
    return maximum, worst


frames = run([(100, {'power': False}), (1300, {'power': True})], 2450)
assert any(not any(wire) for at, wire in frames if 1100 <= at < 1300)
assert any(0 < sum(wire) < 10000 for at, wire in frames if 100 <= at < 1100)
on = [(at, wire) for at, wire in frames if at >= 1300]
# The first ON frame is checked at the driver, before any polling can miss it.
assert 0 < sum(on[0][1]) < sum(on[-1][1]) // 4
print('normal OFF/ON every pixel/frame maximum drift:', accurate(frames))

# Reversing a ramp continues from what's showing, in both directions.
frames = run([(100, {'power': False}), (1300, {'power': True}), (1500, {'power': False}),
              (1700, {'power': True})], 3000)
around = [sum(wire) for at, wire in frames if 1400 <= at <= 1800]
assert all(abs(b - a) < 2000 for a, b in zip(around, around[1:])), around
rising = [sum(wire) for at, wire in frames if 1400 <= at <= 1500]
falling = [sum(wire) for at, wire in frames if 1525 <= at <= 1700]
assert all(a <= b for a, b in zip(rising, rising[1:])), rising
assert all(a >= b for a, b in zip(falling, falling[1:])), falling
assert falling[-1] < falling[0] < 47000 and sum(frames[-1][1]) > 47000

# A command's own duration replaces the default 1 s power and brightness ramps.
frames = run([(100, {'power': False, 'transition': .4}), (900, {'power': True, 'transition': .4})], 1600)
assert any(sum(wire) for at, wire in frames if 400 <= at < 500)
assert all(not any(wire) for at, wire in frames if 550 <= at < 900)
on = [(at, sum(wire)) for at, wire in frames if at >= 900]
assert 0 < on[0][1] < 15000
assert all(n > 47000 for at, n in on if at >= 1350)
assert any(n < 47000 for at, n in on if at >= 1200)
frames = run([(100, {'brightness': 1, 'transition': 2})], 2300)
middle = [sum(wire) for at, wire in frames if 1000 <= at <= 1200]
assert middle and all(15000 < n < 33000 for n in middle), middle
assert sum(frames[-1][1]) < 10000
assert all(sum(wire) > sum(frames[-1][1]) for at, wire in frames if at < 1800)

for style in ('fade', 'wipe', 'dissolve'):
    for interrupted in (False, True):
        commands = [(100, {'effect': 'other', 'transition_style': style, 'transition_s': .5})]
        if interrupted:
            commands.append((200, {'effect': 'third'}))
        commands.extend([(300, {'power': False}),
                         (600, {'power': True, 'effect': 'warm'})])
        frames = run(commands, 1800)
        off = [(at, wire) for at, wire in frames if 300 <= at < 600]
        assert off and all(not any(wire) for at, wire in off), style
        # After OFF, ON has no stale blend and fades through Output again.
        on = [(at, wire) for at, wire in frames if at >= 600]
        assert 0 < sum(on[0][1]) < 10000
        assert sum(on[-1][1]) > 47000
        accurate(on)

# Brightness commands wait for the active blend, then use Output. Include
# an instant override, which must also avoid dimming the old wire snapshot.
for duration in (0, 1):
    frames = run([(100, {'effect': 'third', 'transition_s': .5}),
                  (200, {'brightness': 7, 'transition': duration})], 1800)
    assert all(sum(wire) > 47000 for at, wire in frames if 200 <= at <= 600)
    low = [(at, wire) for at, wire in frames if at >= 1650]
    assert low and all(0 < sum(wire) < 10000 for at, wire in low)
    accurate(low)
print('default-colour wire-frame OFF/ON, interrupted blends and brightness: PASS')
