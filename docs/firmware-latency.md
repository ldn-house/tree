# Firmware responsiveness

`tests/firmware/latency.py` measures commands against the real firmware and effects running on the MicroPython unix port, with mocked hardware and a private local mosquitto. It prints samples and median/max timings; `--out PATH` writes the full samples, configuration, source revision and phase timings as JSON. There are no latency thresholds, and nothing in CI runs it. A run that couldn't measure (a rejected command, missing output, no broker) prints the error, records it as `error` in the JSON report, and exits 1, so a broken measurement never passes for a slow one.

From the repository root, with MicroPython **1.27.0** and mosquitto installed:

```sh
MICROPYTHON=/path/to/micropython-1.27 uv run --with paho-mqtt --with mpy-cross==1.27.0.post2 python tests/firmware/latency.py --out latency-report.json
```

Use `--repeats N` (default 7), `--seed N` (default 0, randomized spacing), and `--mosquitto /path/to/mosquitto` if needed. Run files use a system temporary directory, removed at the end even if measurement fails. `--work-dir PATH` selects a new directory instead (the path must not already exist); `--keep` preserves the run files and prints their path. `--out` writes wherever you specify, independently of run-file cleanup. Without `--out`, results are printed only. The tool does not change Git ignores or require a repository scratch directory. Only loopback HTTP and MQTT are used; there is no real-host option.

## What is timed

Timing starts before the host HTTP request or MQTT publish, including transport and scheduling overhead. A simulator-only probe records the bytes passed to the LED driver. Brightness requires changed state and wire bytes, power requires lit/dark output, and effect switching requires a lit frame from the new effect. Commands are spaced across different task phases.

Visibility is estimated as **driver kick + wire duration + reset gap**: 15 ms for 500 RGB LEDs at 800 kHz, plus 0.3 ms reset. This models a latch rather than observing physical LEDs. The protocol's T0 excludes the reset gap. Host and simulator share a wall clock, which must not jump during the run; 2 ms polling discovers timestamped events without quantizing their measured latency.

Steady brightness output is cross-checked against `GET /frame`. Timed calibration writes directly to the driver without updating the frame returned by that endpoint, so it can show the previous effect throughout playback. The tool records this discrepancy and uses driver output for all timings.

Tracing changes only a temporary bundle. Its allocations and file writes add observer overhead and host scheduling noise. Use the same host, configuration and MicroPython build when comparing revisions; changes to the source layout may require updating the probe (an instrumentation mismatch is reported as an error).

## Main baseline

Measured on 2026-10-24 at `a1f4ceef127a8f3f035e7ebee641fd22d45df128`, macOS, 64-bit MicroPython 1.27.0, 1 MB simulated heap, 500 LEDs, 40 FPS, real bundled effects and default sequence, local mosquitto, seven repetitions per scenario. Dithering is disabled to make the steady brightness byte comparison unambiguous; power limits, brightness floor and colour-floor analysis retain their firmware defaults. This is a measurement of instrumented main, not a board benchmark. Use `--out PATH` to save the full samples from a new run.

| Scenario | Median (ms) | Max (ms) | What bounds it |
|---|---:|---:|---|
| Brightness, MQTT | 37.940 | 41.781 | 25 ms engine cadence + wire transfer |
| Brightness, POST /state | 40.190 | 43.580 | 25 ms engine cadence + wire transfer |
| Locked effect A → B, MQTT | 37.158 | 46.449 | Next engine tick, startup/render + wire transfer |
| Power off, MQTT | 31.331 | 43.200 | Next engine tick + black wire transfer |
| Power on, MQTT | 73.283 | 111.559 | 100 ms off-state polling + startup + wire transfer |
| Calibration lock, POST /calibrate/start | 63.567 | 103.886 | 100 ms calibration polling + black wire transfer |
| Calibration play, HTTP: first dark frame | 34.079 | 65.428 | Cold preparation + 25 ms minimum lead + 0.3 ms reset |
| Calibration play, HTTP: first lit frame | 1034.075 | 1065.431 | Above + 1000 ms sequence dark lead |
| Calibration play, HA: first dark frame | 5007.239 | 5008.751 | Cold preparation + the button's 5000 ms lead (removed in #49) + 0.3 ms reset |
| Calibration play, HA: first lit frame | 6007.235 | 6008.746 | Above + 1000 ms sequence dark lead |
| MQTT command → state received | 96.022 | 197.196 | 200 ms HA publishing loop + local transport |

RGB light commands are skipped: main's `HA._light` only handles state, brightness and effect; it has neither an RGB command handler nor a `solid` effect. Effect switching here is locked `warm_white` → locked `candy_cane`, unloaded and imported anew each repetition. These short activations remain unproven (the engine requires 30 good frames), so effect startup includes the probe-file path; long-running proven activations can avoid it. This pair is a reproducible baseline, not a claim that every effect has the same startup cost.

After #48 (calibration buffers and output stage allocated on lock and play, the sequence decoder imported on the first sequence request, released on unlock), the same campaign was run back to back on `a1f4cee` and on `11d5d88`, seven repetitions each. Every scenario stayed within run-to-run noise: calibration play over HTTP 34.7 → 34.0 ms to the first frame, HA's button 5008.6 → 5003.6 ms, the play call itself 6.0 → 6.1 ms (HTTP) and 6.2 → 2.6 ms (HA), lock 0.2 → 0.1 ms; the lock's first frame, 59.5 → 72.3 ms median with the max 90 → 96 ms, is the 100 ms calibration poll's phase.

Effect import alone measured 0.243 ms median / 0.390 ms max; `Effect.__init__` 0.014 / 0.020 ms. Complete `_start` took 3.100 / 4.994 ms. Most command-to-frame latency is waiting for the next tick and transferring the frame, rather than this effect's import or constructor.

## Why HA calibration waited

Since #49 the button sends `delay_ms: 0`, and with the same tool its first frame arrives in 28.7 ms median / 32.4 ms max and the first lit frame in 1,028.7 / 1,032.4 ms (three repeats), the same as an HTTP play. The rest of this section explains the baseline above. The tool obtains the button configuration from actual retained HA discovery. At `a1f4cee`, “Play calibration sequence” published to `xmas_tree/calibrate/set`:

```json
{"action":"play","sequence":"default","loops":0,"delay_ms":5000,"ttl_s":300}
```

The MQTT handler calls calibration playback directly; it does not wait for HA's state-publishing loop. Playback loads and validates the sequence, takes the calibration lock and prepares lookup tables, then schedules the first latch five seconds in the future. The engine is paused during that lead, leaving the previous frame on the strip. At T0 the first sequence frame is all off for **1000 ms**; the next frame is the first illuminated synchronization pattern. Thus HA's first calibration frame arrives at about five seconds, and the first lit frame at about six seconds. Both delays are intentional filming preparation/settling behavior in `docs/calibration-protocol.md` sections 5, 7 and 9.

| HA calibration phase | Median (ms) | Max (ms) |
|---|---:|---:|
| Send → MQTT handler entered | 1.098 | 1.944 |
| MQTT handler, inclusive | 4.749 | 6.545 |
| Play preparation, inclusive | 4.483 | 6.249 |
| Load file, CRC and frame validation | 2.201 | 2.587 |
| Lock, clear static buffer, take ownership | 0.147 | 0.174 |
| Prewarm mono lookup tables | 1.515 | 2.244 |

The phases are nested, so their medians should not be added. Each playback sample starts after unlock and includes cold sequence loading and lookup-table preparation. The main baseline does not support blaming imports, allocation or MQTT handling for the multi-second wait: preparation takes a few milliseconds. The four calibration frame buffers (6000 payload bytes) are allocated at boot, rather than on the command path.

HTTP `/calibrate/start` only takes the lock and requests a static black frame; it neither loads nor plays a sequence. The calibration task can already be asleep for 100 ms, explaining the slower first blank frame. This incidental polling makes the protocol's claim of static updates within one frame optimistic for this implementation.

## Fixed waits and limitations

- Normal engine cadence: 40 FPS, 25 ms per tick, plus rendering/output work. A previous DMA transfer can add a wait, polled at 1 ms.
- Engine off state: 100 ms polling before it can notice power-on. Calibration ownership: 50 ms polling before normal effects resume.
- Static calibration: 100 ms idle polling (20 ms when still dirty), 1000 ms refresh. Unlock shows black and waits 40 ms before releasing ownership.
- Playback: minimum T0 lead is wire duration plus 10 ms (25 ms here); explicit delay up to 30 s. HA's button requested 5000 ms until #49 and now requests 0. The sequence adds its own 1000 ms dark lead. `_wait_until` yields for the bulk of a deadline and spins for the last roughly 2.5–3.5 ms.
- Completed implicit playback holds its lock for 2 s; this is an **end** delay. Finite playback ignores lock TTL; HA's infinite playback uses a 300 s TTL (general default 180 s, maximum 900 s).
- HA publishes dirty state on a 200 ms task loop. Running-effect publications are limited to one per second, calibration progress is once a second while playing, and diagnostics every 30 s. None gates command acceptance or wire output.
- HA discovery waits up to 1 s for retained availability on connection; birth-triggered rediscovery is delayed 3 s, and online is reasserted after 1.5 s. Measurements begin after discovery completes.
- State persistence is debounced 2 s and deferred during playback; it does not debounce applying state. Supervisor GC is every 10 s, also subject to the runtime's 32 KB allocation threshold; effect startup and calibration preparation collect explicitly. These and flash activity can add incidental stalls. Reserve creation begins after 30 s, repeats every 10 minutes and pauses during playback.
- Colour-floor samples are scheduled at effect times 0, 1, 3 and 6 s, spread with a 4 ms per-frame work budget beyond one colour; a single colour can take about 10 ms on the Pico. The unix timings do not reproduce that CPU cost.

The unix port runs Python much faster than a Pico and has a different allocator/object size, filesystem and network stack. It simulates wire occupancy but not Wi-Fi airtime, real DMA contention, flash erase stalls, camera timing or Home Assistant's own UI-to-broker delay. Relative changes under the same simulator, task/frame-bound delays, and explicit scheduled waits transfer usefully; absolute CPU/import/allocation numbers do not. Seven observed maxima are not upper bounds. One HTTP preparation sample spent 22.955 ms in sequence load/validation (median 2.322 ms), raising that first-frame maximum to 65.428 ms; it is retained as observed host/runtime jitter rather than filtered away. The trace does not distinguish CPU work, filesystem latency, GC and host descheduling within that phase. No real tree was contacted.

Validation exercised all scenarios repeatedly, checked steady output against the public frame API, read the actual discovery command, observed calibration's first blank and first lit output, and checked for firmware effect errors, resets and console exceptions. Adjacent device scenarios also exercised static/sequence calibration, TTL expiry, power-off unlock, MQTT playback, command validation, and running-effect publication on MicroPython 1.27.

## Possible follow-ups

- Wake the off-state and calibration-lock tasks when commands arrive, instead of waiting for 100 ms polling.
- Update `GET /frame` during timed calibration playback so it reflects the output on the strip.
