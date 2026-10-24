"""Measure simulated command-to-wire latency; observational, no latency gates.

MICROPYTHON=... uv run --with paho-mqtt --with mpy-cross==1.27.0.post2 \
    python tests/firmware/latency.py --out latency-report.json

Builds the current real app/effects with simulator-only tracing; uses local
mosquitto and devsim, never a network tree. No pytest dependency. Each run gets
its own temporary directory and ephemeral ports. There are no latency
thresholds; a run that couldn't measure (a rejected command, missing output, no
broker) exits 1, so a broken measurement never passes for a slow one.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import threading
import tempfile
import time
import zlib
from pathlib import Path

import devsim


def now_us():
    return time.time_ns() // 1000


def summary(values):
    return {'median_ms': round(statistics.median(values), 3),
            'max_ms': round(max(values), 3), 'samples_ms': values}


def instrument(dest):
    shutil.copytree(devsim.ROOT / 'firmware/app', dest)
    shutil.copy(Path(__file__).with_name('latency_probe.py'), dest / 'latency_probe.py')

    def patch(name, anchor, replacement):
        p = dest / name
        src = p.read_text()
        if src.count(anchor) != 1:
            raise RuntimeError(f'instrumentation anchor changed: {name}: {anchor!r}')
        p.write_text(src.replace(anchor, replacement))

    patch('runtime.py', 'class Runtime:',
          'from app import latency_probe\nlatency_probe.install()\n\n\nclass Runtime:')
    patch('runtime.py', '        self.sup = supervisor.Supervisor(self)',
          '        latency_probe.rt = self\n        self.sup = supervisor.Supervisor(self)')
    patch('engine.py', '            inst = mod.Effect(ctx)',
          '            from app import latency_probe\n'
          '            latency_probe.event("effect.init.begin")\n'
          '            _latency_started = time.time_ns()\n'
          '            inst = mod.Effect(ctx)\n'
          '            _latency_elapsed = (time.time_ns() - _latency_started) // 1000\n'
          '            latency_probe.event("effect.init.end", duration_us=_latency_elapsed)')


class Bus:
    def __init__(self, port):
        import paho.mqtt.client as mqtt
        self.messages = []
        self.ready = threading.Event()
        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.client.on_connect = lambda c, u, f, rc, p: c.subscribe('#')
        self.client.on_subscribe = lambda *a: self.ready.set()
        self.client.on_message = self.receive
        self.client.connect('127.0.0.1', port)
        self.client.loop_start()
        if not self.ready.wait(10):
            raise TimeoutError('MQTT subscription')

    def receive(self, client, userdata, message):
        self.messages.append((now_us(), message.topic, message.payload, message.retain))

    def send(self, topic, payload):
        self.client.publish(topic, json.dumps(payload)).wait_for_publish(5)

    def close(self):
        self.client.disconnect()
        self.client.loop_stop()


class Campaign:
    def __init__(self, device, bus, report):
        self.d, self.bus, self.report = device, bus, report
        self.trace = device.root / 'latency.jsonl'
        self.offset = 0
        self.events = []

    def read(self):
        if self.trace.exists():
            with self.trace.open() as f:
                f.seek(self.offset)
                while True:
                    line = f.readline()
                    if not line.endswith('\n'):
                        break
                    self.events.append(json.loads(line))
                    self.offset = f.tell()
        return self.events

    def wire(self, after, pred):
        return devsim.wait_for(
            lambda: next((e for e in self.read() if e['event'] == 'wire'
                          and e['kick_us'] >= after and pred(e)), None),
            timeout=15, interval=.002, what='matching wire frame')

    def post(self, path, body=b''):
        code, result = self.d.post(path, body)
        if code != 200:
            raise RuntimeError(f'{path}: {code}: {result}')
        return result

    def prepare(self, **state):
        start = now_us()
        self.post('/state', state)
        return self.wire(start, lambda e: e['calibration'] == 'idle'
                         and not e['owner'] and e['lit']
                         and all(e[k] == v for k, v in state.items() if k in e))

    def restore(self):
        self.post('/calibrate/stop')
        self.prepare(power=True, effect='warm_white', brightness=200)
        # Vary phase relative to the firmware's 25/100/200 ms task ticks.
        time.sleep(random.uniform(.01, .23))

    def measure(self, name, send, pred, *, lit=False, echo=None):
        start = now_us()  # before HTTP request/paho publish, not after its ACK
        send()
        first = self.wire(start, pred)
        metrics = {'first_frame': (first['latch_us'] - start) / 1000}
        if lit:
            visible = self.wire(start, lambda e: pred(e) and e['lit'])
            metrics['first_lit_frame'] = (visible['latch_us'] - start) / 1000
            api_crc = zlib.crc32(base64.b64decode(self.d.get('/frame')[1]['frame']))
            self.report.setdefault('frame_api_observations', []).append({
                'scenario': name, 'matches_first_lit_wire': api_crc == visible['crc']})
        if echo:
            def received():
                return next((at for at, topic, payload, retained in self.bus.messages
                             if at >= start and topic == 'xmas_tree/state' and not retained
                             and echo(json.loads(payload))), None)
            at = devsim.wait_for(received, 10, interval=.002, what='MQTT state echo')
            metrics['state_echo'] = (at - start) / 1000
        events = [e for e in self.read() if start <= e['us'] <= first['us']]
        for e in events:
            if e['event'].endswith('.begin'):
                label = e['event'][:-6]
                end = next((x for x in events if x['event'] == label + '.end'
                            and x['us'] >= e['us']), None)
                if end:
                    metrics[label] = end['duration_us'] / 1000
                metrics[label + '.arrival'] = (e['us'] - start) / 1000
        self.report['samples'].setdefault(name, []).append(metrics)
        print(name, json.dumps(metrics), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repeats', type=int, default=7)
    ap.add_argument('--seed', type=int, default=0, help='repeatable randomized command spacing')
    ap.add_argument('--out', type=Path, help='write full JSON samples to this path')
    ap.add_argument('--work-dir', type=Path, help='use this new directory for run files')
    ap.add_argument('--keep', action='store_true', help='keep run files after completion')
    ap.add_argument('--mosquitto', default=os.environ.get('MOSQUITTO') or shutil.which('mosquitto')
                    or '/opt/homebrew/opt/mosquitto/sbin/mosquitto')
    args = ap.parse_args()
    random.seed(args.seed)
    if args.repeats < 1:
        ap.error('--repeats must be positive')
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix='firmware-latency-'))
    report = {'revision': None,
              'repeats': args.repeats, 'seed': args.seed, 'samples': {}, 'summary': {},
              'measurement': 'simulated Driver.write wire bytes; estimated latch = kick + write_us + RESET_US',
              'clock': 'shared host epoch; MicroPython time.time_ns resolution 1 us; host time.time_ns',
              'poll_ms': 2, 'work_dir': str(root),
              'settings': {'LED_COUNT': 500, 'FPS': 40, 'LED_DITHER': False, 'heap': devsim.HEAP},
              'skipped': {'light_colour': 'main has no RGB light command handler or solid effect'}}
    broker = bus = d = None
    try:
        report['revision'] = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=devsim.ROOT, text=True).strip()
        report['micropython'] = subprocess.check_output(
            [devsim.MICROPYTHON, '--version'], text=True).strip()
        instrument(root / 'app')
        bundle = devsim.build_bundle(root / 'bundle', app=root / 'app', effects=devsim.ROOT / 'effects')
        port = devsim.free_port()
        conf = root / 'mosquitto.conf'
        conf.write_text(f'listener {port} 127.0.0.1\nallow_anonymous true\npersistence false\n')
        broker = subprocess.Popen([args.mosquitto, '-c', str(conf)], stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL)

        def listening():
            try:
                socket.create_connection(('127.0.0.1', port), timeout=.2).close()
                return True
            except OSError:
                return False
        devsim.wait_for(listening, 10, interval=.02, what='local broker')
        bus = Bus(port)
        d = devsim.Device(root / 'device', MQTT_BROKER='127.0.0.1', MQTT_PORT=port, LED_DITHER=False)
        d.install_recovery()
        d.usb_install(bundle)
        d.start()
        d.wait_http(60, pred=lambda s: not s['boot']['trial'] and s['mqtt']['connected'])
        devsim.wait_for(lambda: 'discovery published' in d.console(), 30, what='HA discovery')
        c = Campaign(d, bus, report)
        configs = [json.loads(p) for _, t, p, _ in bus.messages
                   if t.startswith('homeassistant/button/') and p]
        play = next(x for x in configs if x.get('name') == 'Play calibration sequence')
        report['ha_button'] = {k: play[k] for k in ('command_topic', 'payload_press')}
        for _ in range(args.repeats):
            for transport in ('mqtt', 'http'):
                c.restore()
                baseline = c.prepare(brightness=200, effect='warm_white', power=True)
                # Cross-check the tracing endpoint against the public wire API.
                frame = d.get('/frame')[1]['frame']
                if zlib.crc32(base64.b64decode(frame)) != baseline['crc']:
                    raise RuntimeError('driver trace differs from GET /frame')
                time.sleep(random.uniform(.01, .23))
                # The steady real effect must change its actual wire bytes too.
                send = (lambda: bus.send('xmas_tree/set', {'brightness': 80})) if transport == 'mqtt' else (
                    lambda: c.post('/state', {'brightness': 80}))
                c.measure('brightness_' + transport, send,
                          lambda e: e['brightness'] == 80 and e['crc'] != baseline['crc'],
                          echo=(lambda s: s['brightness'] == 80) if transport == 'mqtt' else None)
            c.restore()
            c.measure('effect_mqtt_warm_white_to_candy_cane',
                      lambda: bus.send('xmas_tree/set', {'effect': 'candy_cane'}),
                      lambda e: e['effect'] == 'candy_cane' and e['lit'])
            c.restore()
            c.measure('power_off_mqtt', lambda: bus.send('xmas_tree/set', {'state': 'OFF'}),
                      lambda e: not e['power'] and not e['lit'])
            time.sleep(random.uniform(.01, .23))
            c.measure('power_on_mqtt', lambda: bus.send('xmas_tree/set', {'state': 'ON'}),
                      lambda e: e['power'] and e['lit'] and e['calibration'] == 'idle')
            for name, send in (
                ('calibration_start_http', lambda: c.post('/calibrate/start')),
                ('calibration_play_http', lambda: c.post('/calibrate/play?sequence=default&loops=1')),
                ('calibration_play_ha', lambda: bus.send(play['command_topic'], json.loads(play['payload_press']))),
            ):
                c.restore()
                c.measure(name, send,
                          (lambda e: e['calibration'] == 'locked' and not e['lit']) if name.endswith('start_http')
                          else (lambda e: e['calibration'] == 'playing'), lit=not name.endswith('start_http'))
        status = d.get('/')[1]
        report['effect_errors'] = status['effect']['failed']
        report['boots'] = d.boots
        report['console_errors'] = [line for line in d.console().splitlines()
                                    if 'Traceback' in line or 'MemoryError' in line]
        report['boot_constructor_ms'] = next(e['duration_us'] / 1000 for e in c.read()
                                             if e['event'] == 'calib.construct.end')
    except Exception as exc:
        report['error'] = repr(exc)
        print('Measurement error:', repr(exc), flush=True)
    finally:
        try:
            if d:
                d.stop()
            if bus:
                bus.close()
            if broker:
                broker.terminate()
                broker.wait()
            for name, samples in report['samples'].items():
                report['summary'][name] = {key: summary([s[key] for s in samples if key in s])
                                           for key in sorted({k for s in samples for k in s})}
            print('Summary:', json.dumps(report['summary']), flush=True)
        finally:
            if args.keep:
                print('Run files kept:', root, flush=True)
            else:
                shutil.rmtree(root)
        # Write after cleanup so even an output path inside work_dir survives.
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2) + '\n')
            print('Report:', args.out, flush=True)
        return 1 if 'error' in report else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as exc:
        print('Measurement error:', repr(exc), flush=True)
        sys.exit(1)
