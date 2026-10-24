"""MicroPython transition maths, kernel comparison, allocation and 500 LED timing.

Run with MicroPython 1.27 from the repo root. Native parity is verified only
when NATIVE=True; timings describe the host running this tool, not the Pico.
"""
import gc
import sys
import time
sys.path.insert(0, '.')
sys.path.insert(0, 'firmware')
sys.path.insert(0, 'tests/firmware/mocks')
from app.transitions import Blend, Ramp, NATIVE
from transition_reference import _mix_py
from effects._output import Output
from app.engine import Engine

r = Ramp(0)
for duration in (None, 1000):
    r.set(0, 0, duration)
    assert r.step(0) == 0 and r.step(500) == 0
r = Ramp(255)
r.set(0, 100, 1000)
assert r.step(100) == 255
assert r.step(600) == 128
assert r.step(1099) == 1
assert r.step(1100) == 0
r.set(255, 1100, 1000)
assert r.step(1100) == 1
assert r.step(1600) == 128
r.set(0, 1600, 1000)
assert r.step(1600) == 128
assert r.step(2600) == 0
r = Ramp(100)
r.set(110, 0)
assert r.ms == 39 and r.step(0) == 100 and r.step(39) == 110

one = Blend(1)
one.begin(bytearray([0, 100, 255]), 'fade', 1000, 0)
wire = bytearray([100, 0, 255])
one.apply(wire, 0, 1000)
wire[:] = bytearray([100, 0, 255])
one.apply(wire, 250, 1000)
assert wire == bytearray([25, 75, 255])
one.begin(wire, 'fade', 0, 0)
one.apply(wire, 0, 1000)
assert wire == bytearray([25, 75, 255])

b = Blend(500)
a = bytearray(1500)
c = bytearray(1500)
p = bytearray(1500)
for i in range(1500):
    a[i] = (i * 13) & 255
    c[i] = (i * 7) & 255
for style in ('fade', 'wipe', 'dissolve'):
    b.begin(a, style, 1000, 0)
    for t in (0, 1, 125, 500, 999, 1000):
        p[:] = c
        _mix_py(a, p, b.order, 500, t * 256 // 1000, int(style == 'fade'), 47812)
        wire = bytearray(c)
        b.apply(wire, t, 47812)
        if t < 1000:
            assert wire == p
            assert sum(wire) <= 47812
        else:
            assert wire == c
# Coordinate changes leave an active wipe's table and frame unchanged.
b.begin(a, 'wipe', 1000, 0)
b.apply(p, 0, 47812)
thresholds = bytes(b.order)
b.coords([(0, 1, 0), None, (0, 0, 0)])
assert bytes(b.order) == thresholds
b.apply(p, 1000, 47812)
b.begin(a, 'wipe', 1000, 0)
assert b.order[0] == 255 and b.order[2] == 0 and b.order[1] == 0
b.begin(a, 'none', 1000, 0)
p[:] = c
b.apply(p, 100, 47812)
assert p == c

out = Output(500)
r = Ramp(255)
r.set(1, 0, 1000)
b.begin(a, 'wipe', 1000, 0)
out.process(c, p, 255)
b.apply(p, 500, out.budget)
class Display:
    owner = None
    blend = b
    off_level = 0

display = Display()
for i in range(100):
    Engine.process(display, c, out, p, r.step(i * 9), i * 9)
r.set(255, 0, 1000)
def check_allocations():
    gc.collect()
    gc.disable()
    before = gc.mem_alloc()
    for i in range(100):
        Engine.process(display, c, out, p, r.step(i * 9), i * 9)
    allocated = gc.mem_alloc() - before
    gc.enable()
    assert allocated == 0, allocated

check_allocations()
for style in ('fade', 'wipe', 'dissolve'):
    b.begin(a, style, 1000, 0)
    b.apply(p, 0, out.budget)
    t = time.ticks_us()
    for i in range(1000):
        r.step(500)
        b.apply(p, 500, out.budget)
    us = time.ticks_diff(time.ticks_us(), t) // 1000
    t = time.ticks_us()
    for i in range(1000):
        Engine.process(display, c, out, p, r.step(500), 500)
    total_us = time.ticks_diff(time.ticks_us(), t) // 1000
    print(style, us, 'us blend/ramp;', total_us, 'us output + blend/ramp; native:', NATIVE)
# The frozen OFF path also must allocate nothing and preserve wire hue.
b.begin(bytearray((68, 23, 4) * 500), 'none', 0, 0)
display.off_level = 65280
out.set_floor(25)
r = Ramp(255)
r.set(1, 0, 1000)
check_allocations()
Engine.process(display, c, out, p, 1, 0)
assert p[:3] == bytearray((17, 6, 1))
assert all(p[k] <= b.old[k] for k in range(len(p)))
# Check the selected dim kernel against independent upward-rounding maths.
b.old[:] = a
for level in (0, 1, 256, 16320, 32768, 65279, 65280):
    b.dim(p, level, 65280)
    q = min(16384, level * 16384 // 65280)
    assert all(p[k] == (b.old[k] * q + 16383) // 16384 for k in range(len(p)))
t = time.ticks_us()
for i in range(1000):
    Engine.process(display, c, out, p, r.step(500), 500)
print('frozen OFF', time.ticks_diff(time.ticks_us(), t) // 1000, 'us output + ramp; native:', NATIVE)
print('kernel comparison:', 'native vs Python PASS' if NATIVE else 'Python vs reference PASS; native parity NOT VERIFIED')
print('transition maths, output + ramp + blend + frozen OFF allocation: PASS')
