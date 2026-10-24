"""Lifecycle and state regressions for the transitions review, on MicroPython."""
import os
import sys

repo = sys.argv[1]
sys.path.insert(0, repo + '/firmware')
sys.path.insert(0, repo + '/tests/firmware/mocks')
sys.path.insert(0, repo)
os.chdir(sys.argv[2])
from app.state import State
from app.engine import Engine
from effects._output import Output

s = State()
events = []
s.listeners.append(lambda event, source: events.append(event))
s.update({'brightness': 1, 'transition': 60})
s._dirty_at = None
assert s.update({'brightness': 1, 'transition': 0}) == {}
assert events[-1] == {'brightness': 1, 'transition_ms': 0}
assert s._dirty_at is None
s.update({'power': False, 'transition': 60})
s._dirty_at = None
assert s.update({'power': False, 'transition': 0}) == {}
assert events[-1] == {'power': False, 'transition_ms': 0}
assert s._dirty_at is None

class Effect:
    def __init__(self, ctx):
        pass

class Module:
    Effect = Effect

class FX:
    def names(self):
        return ['bad', 'good']
    def load(self, name):
        return Module
    def unload(self, name):
        pass
    def meta(self, name):
        return {'params': {}}

s = State()
e = Engine(1, s, FX(), Output(1), None, 40)
# Failure after a completed blend, and failure during an active blend,
# must both snapshot the last displayed frame, even after _fail unloads.
for active in (False, True):
    e.name = 'bad'
    e.inst = Effect(None)
    e.displayed = True
    e.last[:] = b'\x20\x10\x08'
    e.blend.begin(b'\xff\xff\xff', 'fade', 1000 if active else 0, 0)
    e._fail('bad', ValueError('synthetic render failure'))
    assert e.inst is None
    assert e._start('good')
    assert e.blend.old == e.last
    assert e.blend.pending
    wire = bytearray(b'\x00\x00\xff')
    e.blend.apply(wire, 10000, 765)
    assert wire == e.last  # first replacement frame starts at the snapshot
    assert e.blend.ms == 500
# Boot-time coordinates also reach the first spatial transition.
e.coords = [(0, 1, 0)]
e.transition('wipe', 1)
assert e.blend.order[0] == 255
# Calibration owns exact output and never starts a replacement blend.
e.owner = object()
e.blend.ms = 0
assert e._start('good')
assert e.blend.ms == 0
print('transition lifecycle and repeated-command regressions: PASS')
