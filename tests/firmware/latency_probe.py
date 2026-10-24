"""Simulator-only instrumentation, copied into a private app bundle by latency.py.

Never imported by production firmware. Wall-clock timestamps share the host's
clock on the unix port (time.time_ns has microsecond resolution there).
"""
import binascii
import json
import time

rt = None


def event(kind, **fields):
    fields['event'] = kind
    fields['us'] = time.time_ns() // 1000
    with open('latency.jsonl', 'a') as f:
        f.write(json.dumps(fields) + '\n')


def wrap(cls, method, label):
    original = getattr(cls, method)

    def measured(self, *args, **kwargs):
        event(label + '.begin')
        started = time.time_ns()
        try:
            return original(self, *args, **kwargs)
        finally:
            elapsed = (time.time_ns() - started) // 1000
            event(label + '.end', duration_us=elapsed)

    setattr(cls, method, measured)


def install():
    from app import calib, driver, engine, fx, ha

    for cls, method, label in (
        (calib.Calib, '__init__', 'calib.construct'),
        (calib.Calib, 'lock', 'calib.lock'),
        (calib.Calib, 'default_sequence', 'calib.load_validate'),
        (calib.Calib, 'play', 'calib.play'),
        (calib.Sequence, 'prewarm', 'calib.prewarm'),
        (fx.Effects, 'load', 'effect.import'),
        (engine.Engine, '_start', 'effect.start'),
        (ha.HA, '_light', 'mqtt.light'),
        (ha.HA, '_calibrate', 'mqtt.calibrate'),
    ):
        wrap(cls, method, label)

    original = driver.Driver.write

    def write(self, buf):
        # Timestamp before instrumentation; estimate visibility at end of wire
        # transfer plus reset gap. The DMA mock itself copies at transfer start.
        kick = time.time_ns() // 1000
        original(self, buf)
        c = rt.calib
        event('wire', kick_us=kick, latch_us=kick + self.busy_us,
              crc=binascii.crc32(buf) & 0xffffffff, lit=bool(any(buf)),
              effect=rt.engine.name, power=rt.state.get('power'),
              brightness=rt.state.get('brightness'), calibration=c.state,
              owner=rt.engine.owner is not None,
              frame=c.play_info['frame'] if c.play_info else None)

    driver.Driver.write = write
