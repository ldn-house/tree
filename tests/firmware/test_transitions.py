"""Display transitions observed from the simulated driver's wire frames."""
import base64
import json
import subprocess
import time

import devsim
import pytest
from devsim import wait_for
from test_mqtt import mdev  # MQTT fixture


def frame(d):
    return base64.b64decode(d.get('/frame')[1]['frame'])


def level(d):
    return sum(frame(d))


def setup(d, **extra):
    values = dict(effect='solid', power=True, brightness=255, colour_accuracy=0,
                  transition_style='none', transition=0)
    values.update(extra)
    assert d.post('/state', values)[0] == 200
    return wait_for(lambda: (lambda n: n if n > 47000 else None)(level(d)), 5)


def samples(d, seconds):
    result = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        result.append((time.monotonic(), level(d)))
        time.sleep(.025)
    return result


def test_transition_maths_allocations_and_timing():
    r = subprocess.run([devsim.MICROPYTHON, str(devsim.ROOT / 'tools/transition_check.py')],
                       cwd=devsim.ROOT, capture_output=True, text=True)
    assert r.returncode == 0 and 'PASS' in r.stdout, r.stdout + r.stderr


def test_effect_blend_and_interrupt(device):
    d = device
    setup(d)
    # Green/blue height effect has no red: intermediate frames must keep
    # some of the frozen white frame's red, and acquire the new colours.
    d.post('/state', {'transition_style': 'fade', 'transition_s': 1, 'effect': 'height'})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 5)
    f = frame(d)
    assert sum(f[0::3]) > 0 and f != bytes([f[0]]) * 1500
    # Interrupt: a new effect freezes the currently blended wire frame.
    before = frame(d)
    d.post('/state', {'effect': 'solid'})
    after = frame(d)
    assert abs(sum(after) - sum(before)) < 5000
    wait_for(lambda: abs(level(d) - 47812) < 100, 3)
    d.post('/state', {'effect': 'height'})
    wait_for(lambda: sum(frame(d)[0::3]) == 0, 3)
    assert d.get('/')[1]['effect']['failed'] == {}


def test_none_and_zero_cut_and_auto_blends(device):
    d = device
    setup(d)
    for values in ({'transition_style': 'none', 'transition_s': 60},
                   {'transition_style': 'fade', 'transition_s': 0}):
        d.post('/state', dict(values, effect='height'))
        wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 3)
        wait_for(lambda: sum(frame(d)[0::3]) == 0, .3)
        d.post('/state', {'effect': 'solid'})
        wait_for(lambda: level(d) > 47000, 3)
    d.post('/state', {'effect': 'auto', 'auto_s': 1, 'transition_style': 'fade', 'transition_s': .5})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'chase', 5)
    assert sum(frame(d)[1::3]) > 0  # frozen white survives the switch
    wait_for(lambda: sum(frame(d)[1::3]) == 0, 1)


def test_power_fades_floor_and_reversal(device):
    d = device
    full = setup(d)
    d.post('/state', {'power': False})
    down = samples(d, 1.2)
    assert all(b[1] <= a[1] for a, b in zip(down, down[1:]))
    assert 0 < down[0][1] <= full and down[-1][1] == 0
    assert any(0 < n < full // 2 for _, n in down)
    d.post('/state', {'power': True})
    wait_for(lambda: level(d) > 0, .3)
    up = samples(d, .45)
    assert all(b[1] >= a[1] for a, b in zip(up, up[1:]))
    assert 1500 <= up[0][1] < full // 4  # effective floor, never sub-floor
    before = level(d)
    d.post('/state', {'power': False})
    after = level(d)
    assert abs(before - after) < 2000
    wait_for(lambda: level(d) == 0, 2)
    d.post('/state', {'power': True})
    wait_for(lambda: level(d) > full - 100, 2)
    # Brightness-only ramps also go through the floor, without analysis reset.
    d.post('/state', {'brightness': 1})
    vals = samples(d, 1.15)
    assert all(b[1] <= a[1] for a, b in zip(vals, vals[1:]))
    assert vals[-1][1] == 1500
    d.post('/state', {'power': False, 'transition': 0})
    wait_for(lambda: level(d) == 0, 1)
    d.post('/state', {'brightness': 77})
    assert all(n == 0 for _, n in samples(d, .3))


def test_ha_duration_discovery_and_persistence(mdev, bus):
    d = mdev
    setup(d)
    cfg = json.loads(bus.wait('homeassistant/light/xmas_tree/config'))
    assert cfg['transition'] is True
    cfg = json.loads(bus.wait('homeassistant/select/xmas_tree/transition_style/config'))
    assert cfg['options'] == ['none', 'fade', 'wipe', 'dissolve']
    assert json.loads(bus.wait('homeassistant/number/xmas_tree/transition_s/config'))['unit_of_measurement'] == 's'
    bus.pub('xmas_tree/set', '{"brightness":1,"transition":2}')
    wait_for(lambda: d.get('/')[1]['state']['brightness'] == 1, 3)
    wait_for(lambda: json.loads(bus.last('xmas_tree/state'))['brightness'] == 1, 3)
    vals = samples(d, 2.2)
    start = vals[0][0]
    middle = [n for t, n in vals if .8 <= t - start <= 1.1]
    assert middle and all(15000 < n < 33000 for n in middle)
    assert vals[-1][1] == 1500
    bus.pub('xmas_tree/cmd/transition_style', 'wipe')
    bus.pub('xmas_tree/cmd/transition_s', '1.5')
    wait_for(lambda: d.saved_state().get('transition_s') == 1.5, 6)
    assert d.saved_state()['transition_style'] == 'wipe'
    assert 'transition' not in d.saved_state() and 'transition_ms' not in d.saved_state()
    d.kill_now('pwron')
    st = d.wait_http(30, pred=lambda s: len(d.boots) >= 1)
    assert st['state']['transition_style'] == 'wipe' and st['state']['transition_s'] == 1.5


def test_calibration_bypasses_active_blend(device):
    d = device
    setup(d)
    d.post('/state', {'effect': 'height', 'transition_style': 'fade', 'transition_s': 60})
    d.get('/calibrate/start', token=True)
    d.get('/calibrate/led?n=7', token=True)
    f = wait_for(lambda: (lambda f: f if f[21:24] == b'\xff\xff\xff' else None)(frame(d)), 3)
    assert sum(f) == 765
    d.get('/calibrate/stop', token=True)
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 3)
    assert sum(frame(d)[0::3]) == 0


@pytest.mark.parametrize('style', ['wipe', 'dissolve'])
def test_spatial_blend_wire_frames_and_cap(device, style):
    d = device
    setup(d)
    d.post('/state', {'effect': 'height', 'transition_style': style, 'transition_s': 1})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 3)
    vals = []
    end = time.monotonic() + 1.3
    while time.monotonic() < end:
        f = frame(d)
        vals.append(sum(f[0::3]))
        assert sum(f) <= 47812
        time.sleep(.025)
    assert any(0 < n < 15000 for n in vals) and vals[-1] == 0
    assert d.post('/state', {'transition_style': 'bogus'})[0] == 400
    assert d.post('/state', {'transition_s': 'x'})[0] == 400
    assert d.post('/state', {'transition': -1})[0] == 200  # clamped to instant


def test_small_brightness_change_settles_quickly(device):
    d = device
    setup(d)
    d.post('/state', {'brightness': 128, 'transition': 0})
    mid = wait_for(lambda: (lambda n: n if 23500 < n < 25500 else None)(level(d)), 1)
    d.post('/state', {'brightness': 138})
    # Ten slider units take 39 ms plus frame pacing; no one-second tail.
    wait_for(lambda: level(d) > mid + 1200, .25, interval=.01)


def test_ha_power_transition_override(mdev, bus):
    d = mdev
    setup(d)
    bus.pub('xmas_tree/set', '{"state":"OFF","transition":0.4}')
    wait_for(lambda: d.get('/')[1]['state']['power'] is False, 2, interval=.01)
    assert level(d) > 0
    wait_for(lambda: level(d) == 0, .65, interval=.01)
    bus.pub('xmas_tree/set', '{"state":"ON","transition":0.4}')
    wait_for(lambda: level(d) > 0, .3, interval=.01)
    assert level(d) < 15000
    wait_for(lambda: level(d) > 47000, .65, interval=.01)


def test_power_off_during_long_effect_blend(device):
    d = device
    setup(d)
    d.post('/state', {'effect': 'height', 'transition_style': 'fade', 'transition_s': 60})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 3)
    before = level(d)
    d.post('/state', {'power': False})
    assert abs(level(d) - before) < 2000
    vals = samples(d, 1.3)
    assert vals[-1][1] == 0
    assert any(0 < n < before // 2 for _, n in vals)


def test_off_during_dark_to_light_blend(device):
    d = device
    setup(d)
    d.post('/state', {'params': {'solid': {'level': 0}}})
    wait_for(lambda: level(d) == 0, 2)
    d.post('/state', {'effect': 'height', 'transition_style': 'fade', 'transition_s': 60})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'height', 2)
    before = level(d)
    d.post('/state', {'power': False})
    vals = samples(d, 1.3)
    print('OFF dark-to-light: before=', before, 'peak=', max(n for _, n in vals), 'last=', vals[-1][1])
    assert max(n for _, n in vals) <= before + 1000


def test_same_target_can_finish_transition(device):
    d = device
    setup(d)
    d.post('/state', {'brightness': 1, 'transition': 60})
    wait_for(lambda: level(d) < 47000, 3)
    d.post('/state', {'brightness': 1, 'transition': 0})
    vals = samples(d, .3)
    print('same-target instant:', [n for _, n in vals])
    assert vals[-1][1] == 1500


def test_off_blend_respects_floor(device):
    from test_runtime import _put, GOOD
    d = device
    setup(d, floor=25)
    d.post('/state', {'params': {'solid': {'tint': 1}}, 'brightness': 1, 'transition': 0})
    floor_frame = wait_for(lambda: (lambda f: f if f[0] > f[1] else None)(frame(d)), 2)
    floor_sum = sum(floor_frame)
    d.post('/state', {'brightness': 255, 'transition': 0})
    wait_for(lambda: level(d) > 47000, 2)
    black = GOOD.replace(b'"blue"', b'"black"').replace(b'(100, 0, 255)', b'(0, 0, 255)')
    assert _put(d, 'black', black, '')[0] == 200
    d.post('/state', {'effect': 'black', 'transition_style': 'fade', 'transition_s': 60})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'black', 2)
    d.post('/state', {'power': False})
    observed = []
    end = time.monotonic() + 1.3
    while time.monotonic() < end:
        f = frame(d)
        observed.append((sum(f), tuple(f[:3])))
        time.sleep(.02)
    low = [(n, rgb) for n, rgb in observed if 0 < n < floor_sum]
    print('floor=', floor_sum, tuple(floor_frame[:3]), 'sub-floor=', low)
    assert not low
    assert all(rgb[0] >= 17 and rgb[1] >= 6 and rgb[2] >= 1
               for n, rgb in observed if n)
    assert observed[-1][0] == 0


def test_slow_start_preserves_first_blend_frame(device):
    from test_runtime import _put, GOOD
    d = device
    setup(d)
    slow = GOOD.replace(b'"blue"', b'"slowblue"').replace(b'self.ctx = ctx', b'import time\n        time.sleep_ms(600)\n        self.ctx = ctx')
    assert _put(d, 'slowblue', slow, '')[0] == 200
    d.post('/state', {'effect': 'slowblue', 'transition_style': 'fade', 'transition_s': .5})
    wait_for(lambda: d.get('/')[1]['effect']['running'] == 'slowblue', 2, interval=.01)
    f = frame(d)
    print('first slow-start frame red sum=', sum(f[::3]), 'blue=', sum(f[2::3]))
    assert sum(f[::3]) > 0


def test_transition_review_lifecycle_and_same_target_state(tmp_path):
    r = subprocess.run([devsim.MICROPYTHON,
                        str(devsim.ROOT / 'tests/firmware/mp/transitions_review.py'),
                        str(devsim.ROOT), str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0 and 'PASS' in r.stdout, r.stdout + r.stderr
