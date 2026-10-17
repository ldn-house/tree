# Calibration protocol

Status: **draft v1**. Owner: the calibration thread (`calibration/video-pipeline`). Consumer: the firmware rewrite (`firmware/rewrite`).

This document says exactly what the firmware must do so that the tree can play a calibration sequence that a phone (or webcam) films and an offline decoder turns into 3D LED coordinates. The key words MUST, SHOULD and MAY are used as in RFC 2119.

## 1. Division of labour

The firmware is a **dumb, precisely timed frame player**. It does not know about Gray codes, parity, sync markers or colour multiplexing. The host builds a sequence of frames, the firmware plays them at the requested times, and the decoder (`calibrate.py decode-video`) works out everything else from the video.

This keeps the firmware small and lets the encoding change (more redundancy, colour multiplexing, a different sync preamble) without a firmware release. The standard sequence layout is described for information in section 9, but nothing in it is a firmware requirement.

| Concern | Owner |
|---|---|
| Sequence file format, validation, storage | firmware (this spec) |
| Frame timing, output stage, power cap | firmware (this spec) |
| Locking the tree while calibrating, restoring state afterwards | firmware (this spec) |
| HTTP and MQTT control surface | firmware (this spec) |
| Pattern design, sync, codes, levels under the power budget | host (`calib/sequence.py`) |
| Decoding video, triangulation, writing `coords.txt` | host (`calib/`) |

## 2. Calibration session state machine

```
           lock / play                      play
  IDLE  ───────────────▶  LOCKED  ───────────────────▶  PLAYING
   ▲                      │   ▲  ◀──── sequence ends ───┘  │
   │       unlock /       │   └──────── halt ──────────────┘
   └────── TTL expiry ────┘
          (from LOCKED or PLAYING; HA power-off also unlocks)
```

- **IDLE**: normal operation.
- **LOCKED**: the effect scheduler is suspended and the strip shows the current static calibration frame (all off by default). MQTT stays connected and serviced. Home Assistant commands are handled as described in section 7.3.
- **PLAYING**: a sequence is being played (section 5). PLAYING implies LOCKED.

Rules:

1. Entering LOCKED MUST stop the running effect within one frame and show all-off.
2. The lock has a time-to-live (TTL). Default 180 s, maximum 900 s. Every `/calibrate/*` HTTP request and every `xmas_tree/calibrate/set` message refreshes it. While PLAYING, the TTL MUST NOT expire before the current playback finishes, unless the playback loops forever (`loops = 0`), in which case the TTL applies. When the TTL expires the firmware MUST unlock (this fixes review item M7).
3. Unlocking (explicitly, by TTL, or by HA power-off) MUST stop playback, turn the strip off for at least one frame, and then restore the pre-lock `{power, brightness, effect}` state, including any HA changes deferred while locked.
4. `play` while IDLE MUST take an implicit lock. That lock ends automatically 2 s after the playback ends (or at TTL expiry for infinite loops), so a one-shot trigger from HA or curl leaves the tree as it was. Requests do not extend an implicit lock (polling `GET /calibrate` while it plays has no effect on when it ends). They do refresh an explicit lock's TTL.
5. The lock MUST NOT persist across a reboot: the tree always boots IDLE.

## 3. Output path

Calibration frames MUST go through the same runtime output stage as effects (`docs/effect-api.md`, "Output stage"), with two calibration-specific settings:

- **Brightness is 1.0 and gamma is off.** Frame levels are absolute linear drive values. The decoder relies on code frames having the level the file says, regardless of the user's brightness slider.
- **The power cap still applies.** If a frame's raw RGB sum exceeds the budget, the stage scales it uniformly, exactly as for effects. The host builds standard sequences to stay under the budget reported by `GET /calibrate` (section 6), so in practice the cap never engages. When it does engage, the firmware MUST count it in the playback log.
- **Colour order** is the stage's job: levels in the file are logical R, G, B.

Today's firmware writes calibration frames directly and bypasses the cap (review M6). The rewrite MUST NOT do this.

## 4. Sequence file format (`.tcal`, version 1)

All integers are little-endian. The reference encoder and decoder are in [`calib/seqfile.py`](../calib/seqfile.py). It uses only `struct` and `binascii`, runs unmodified under MicroPython, and can be copied into the firmware.

```
header (16 bytes)
  off  type  field
  0    4s    magic           b"TCAL"
  4    u8    version         1
  5    u8    flags           0 (reserved; reject non-zero)
  6    u16   led_count       MUST equal the configured LED count (500)
  8    u16   frame_count     >= 1
  10   u16   default_loops   0 = loop until halted/unlocked/TTL
  12   u32   reserved        0

frame (frame_count times)
  0    u16   duration_ms     hold time, 1..65535
  2    u8    frame_flags     bit0 MONO; other bits reserved (reject non-zero)
  3    u8    level_r
  4    u8    level_g
  5    u8    level_b
  6    ...   planes          MONO: 1 plane; otherwise 3 planes (R, G, B)
                             each plane is ceil(led_count / 8) bytes (63 for 500)

trailer (4 bytes)
  0    u32   crc32           binascii.crc32 of every preceding byte
```

**Bitplanes.** LED `i` is bit `(i & 7)` of byte `(i >> 3)`, least significant bit first. Pad bits past `led_count` MUST be ignored.

**Expanding a frame** into logical RGB for LED `i`:

- MONO: if the bit is set, the LED is `(level_r, level_g, level_b)`, otherwise `(0, 0, 0)`.
- Three planes: channel `c` is `level_c` if bit `i` of plane `c` is set, otherwise 0.

**Validation.** The firmware MUST reject a file (HTTP 400, with the reason in the JSON body) if the magic, version, flags, CRC, total length or `led_count` are wrong, if any `duration_ms` is below `min_frame_ms`, or if the file is larger than `max_sequence_bytes`. It SHOULD compute each frame's raw RGB sum on upload and report any frames over the power budget (they are accepted, and the cap will scale them).

**Limits for sequence producers.** A sequence SHOULD use at most 4 distinct `(level_r, level_g, level_b)` triples across its MONO frames. The firmware expands MONO frames from one 6 KB lookup table per distinct level and keeps at most 4, built before `T0`; more than that forces rebuilds during playback, which can make frames late. Colour (three-plane) frames are expanded per LED without tables. Their cost on the RP2350 is not yet measured against `min_frame_ms`. The standard mono sequence uses 2 levels and no colour frames.

**Sizes.** A MONO frame is 69 bytes and a colour frame is 195 bytes. The standard sequences are 3 to 5 KB. A one-LED-at-a-time sequence of 530 frames is about 37 KB. The firmware MUST accept files of at least 48 KB and SHOULD accept 64 KB. It MAY keep the file in flash and expand frames from it lazily (`seqfile.frame_offsets` shows the access pattern).

**Test vectors.** `tests/data/calib-default.tcal` is the standard sequence. `tests/data/calib-default.expanded.sha256` is the SHA-256 of every frame expanded to `bytearray(3 * 500)` in logical RGB and concatenated in order. `tests/test_seqfile.py` checks both, and the firmware SHOULD check the same hash on the device. The CRC-32 is the standard zlib/PNG one, so `binascii.crc32` on MicroPython and `zlib.crc32` on CPython agree.

## 5. Playback timing

**Definitions.** A frame is *shown* at its **latch time**, the moment the WS2811 data for that frame finishes and the strip latches, which is the end of the strip write. `T0` is the latch time of frame 0 of loop 0. Frame `k` of loop `L` has a scheduled latch time:

```
T(L, k) = T0 + L * loop_ms + sum(duration_ms[0..k-1])
loop_ms = sum(duration_ms)
```

Requirements:

1. The firmware MUST schedule against absolute deadlines (`time.ticks_us()` relative to `T0`), not against chained sleeps, so error does not accumulate across a sequence.
2. Each latch MUST happen within **±4 ms** of its scheduled time, and SHOULD within ±2 ms, for any frame whose duration is at least `min_frame_ms`.
3. A late frame MUST still be shown (it is held for less time), and later frames keep their original deadlines. If a frame is already overdue by more than its own duration when its turn comes, it MAY be skipped. Either way it MUST be recorded in the log.
4. Loops play back-to-back with no gap. Loop `L+1` frame 0 is scheduled at `T0 + (L+1) * loop_ms`.
5. When playback ends, the strip MUST show all-off and the state returns to LOCKED (or to IDLE, for an implicit lock, after 2 s).
6. Playback MUST NOT starve the watchdog, and MUST keep MQTT keepalives going. The firmware SHOULD defer non-essential MQTT publishes and Wi-Fi housekeeping until the gap after a latch.
7. `T0` is `delay_ms` after the `play` request is accepted (default 0, max 30000). This gives a person time to start recording. Frame 0 needs its write time on the wire before it can latch, so the firmware enforces `T0 >= now + write_us + 10 ms` (25 ms for 500 LEDs). It honours any larger `delay_ms` exactly, and reports the actual lead as `t0_in_ms`.

**Rate.** With PIO + DMA the firmware starts each transfer at `due - write_us`, where `write_us` is 500 × 30 µs plus the reset gap, and logs latch error against the DMA-completion estimate. The firmware MUST sustain `min_frame_ms <= 33` (30 fps) with 500 LEDs and SHOULD sustain `min_frame_ms <= 20` (50 fps). It reports the value it actually supports in `GET /calibrate`, and the host never builds frames shorter than that. The standard sequences use 66–100 ms frames because the phone camera is the bottleneck (see the findings report), so the firmware rate limit is a ceiling, not the working point.

**Budget on a Pico 2W.** A 500-LED write is 500 × 24 bits × 1.25 µs = 15 ms on the wire whatever the driver. With today's bit-banged `neopixel` that time is spent with IRQs disabled (review M2). That still fits 30 fps but stalls Wi-Fi. With a PIO + DMA driver the write runs in the background, which is the recommended design. Expanding the next frame SHOULD happen during the current frame's hold time, so that only a buffer swap and DMA kick remain at the deadline. Two cheap ways to expand:

- **MONO frames via a lookup table.** Build `LUT[byte] -> 24-byte chunk` once per distinct level (256 × 24 = 6 KB). Then the frame is 63 slice copies (`buf[24*j : 24*j+24] = LUT[plane[j]]`), well under 1 ms.
- **Colour frames.** A `@micropython.viper` loop over 500 LEDs testing three bits each. On RP2350 this is expected to take about 1 ms, against an estimated 3–5 ms for the pure-Python reference `Frame.rgb()`.

A suggested asyncio structure:

```python
async def play(seq, loops, t0_us):
    k = 0; loop = 0; due = t0_us
    nxt = expand(seq, 0)                      # frame 0 ready before T0
    while loops == 0 or loop < loops:
        await sleep_until(due - WRITE_US)    # wake early by the write time
        driver.show(nxt)                     # latch ~= due
        log.record(k, ticks_diff(ticks_us(), due))
        due += seq.duration_us[k]
        k += 1
        if k == seq.frame_count:
            k = 0; loop += 1
        nxt = expand(seq, k)                  # during this frame's hold
    driver.show(OFF)
```

`sleep_until` is `await asyncio.sleep_ms(...)` for the bulk of the wait, followed by a short busy-wait on `ticks_us()` for the last 1–2 ms.

## 6. HTTP API

These endpoints live on the existing control server (port 8080). All responses are JSON with `Content-Type: application/json`. Errors use `{"error": "..."}` with 400 (bad request), 401 (missing or wrong token), 409 (wrong state) or 413 (too large).

**Authentication.** Every mutating endpoint requires the shared HTTP token from `config.py` (`HTTP_TOKEN`), sent as `Authorization: Bearer <token>` or `X-Tree-Token: <token>`. That covers `/calibrate/start`, `stop`, `sequence`, `play`, `halt` and `frame`, and the legacy `led`, `binary` and `clear`. A missing or wrong token gets 401. The read-only endpoints `GET /calibrate`, `/calibrate/status` and `/calibrate/log` stay open. Host tools take the token from the `TREE_TOKEN` environment variable, then from `config.py`'s `HTTP_TOKEN` (`capture.py` does this in `pico_request`). MQTT commands need no token; broker authentication covers them.

**CORS.** Every `/calibrate*` response MUST include `Access-Control-Allow-Origin: *`. The server MUST answer `OPTIONS` preflight requests with `Access-Control-Allow-Methods: GET, POST, OPTIONS`, `Access-Control-Allow-Headers: Content-Type, Authorization, X-Tree-Token` and `Access-Control-Allow-Private-Network: true`. This lets a browser page on another origin drive calibration (section 10 explains the remaining mixed-content limits).

| Endpoint | Method | Body | Effect |
|---|---|---|---|
| `/calibrate` | GET | – | Status and capabilities (below). Refreshes the TTL if locked. |
| `/calibrate/start?ttl_s=N` | POST (GET accepted for legacy) | – | Lock. Idempotent: if already locked, it only refreshes the TTL. |
| `/calibrate/stop` | POST (GET accepted for legacy) | – | Unlock and restore (section 2, rule 3). |
| `/calibrate/sequence` | POST | `.tcal` bytes | Validate and store as the `uploaded` sequence (RAM). Allowed in any state except PLAYING (409). |
| `/calibrate/play?sequence=uploaded\|default&loops=N&delay_ms=D` | POST | – | Start playback (implicit lock if IDLE). `loops` defaults to the file's `default_loops`. 409 if already PLAYING. |
| `/calibrate/halt` | POST | – | Stop playback, show all-off, stay LOCKED. |
| `/calibrate/frame` | POST | `.tcal` with `frame_count = 1` | Show one static frame while LOCKED (for still-camera capture). `duration_ms` is ignored. 409 if not LOCKED. |
| `/calibrate/log` | GET | – | Timing log of the last or current playback (below). |

**Legacy endpoints.** These are used by `capture.py` and SHOULD stay, implemented as static frames through the output stage:

- `GET /calibrate/led?n=X` lights LED X alone (n = -1 means all off).
- `GET /calibrate/binary?bit=X` lights the LEDs whose index has bit X set.
- `GET /calibrate/clear` turns everything off.
- `GET /calibrate/status` returns the old shape.

Legacy levels: `/calibrate/led` used (255, 255, 255) and `/calibrate/binary` used (80, 80, 80). Under the output stage, `binary` exceeds the cap and gets scaled, which is fine.

`GET /calibrate` response:

```json
{
  "protocol": 1,
  "state": "idle",
  "led_count": 500,
  "power_budget": 49725,
  "min_frame_ms": 20,
  "max_sequence_bytes": 65536,
  "ttl_s": 0,
  "sequences": {
    "default": {"crc32": "1a2b3c4d", "frames": 57, "loop_ms": 6600},
    "uploaded": null
  },
  "playing": null
}
```

- `power_budget` is the raw RGB sum allowed per frame.
- `ttl_s` is the number of seconds left on the lock (0 when IDLE).
- While PLAYING, `playing` is `{"sequence": "default", "loop": 1, "loops": 0, "frame": 23, "t0_ms_ago": 8123}`.

`POST /calibrate/sequence` response: `{"crc32": "...", "frames": 57, "loop_ms": 6600, "over_budget_frames": []}`.

`POST /calibrate/play` response: `{"sequence": "default", "loops": 0, "loop_ms": 6600, "t0_in_ms": 3000}`.

`GET /calibrate/log` response:

```json
{
  "sequence": "default", "frames_shown": 171, "frames_skipped": 0,
  "late_us": {"max": 1830, "mean": 410, "over_4ms": 0},
  "capped_frames": 0,
  "worst_frame": [1, 23]
}
```

`worst_frame` is `[loop, frame]` for the largest latch error. The firmware MAY add `"late_us_by_frame": [...]` for the current or last loop, with -1 marking a skipped frame.

## 7. MQTT and Home Assistant

### 7.1 Topics

- `xmas_tree/calibrate/set` (command, not retained). JSON:
  - `{"action": "play", "sequence": "default", "loops": 0, "delay_ms": 5000, "ttl_s": 300}`. Every field except `action` is optional.
  - `{"action": "halt"}`
  - `{"action": "unlock"}`
- `xmas_tree/calibrate/state` (retained, published on every state change and once a second while PLAYING):

```json
{"state": "playing", "sequence": "default", "loop": 2, "loops": 0, "loop_ms": 6600, "ttl_s": 241}
```

### 7.2 Discovery

The firmware SHOULD publish these entities, attached to the existing tree device:

- `button` "Play calibration sequence": command topic `xmas_tree/calibrate/set`, `payload_press` `{"action":"play","sequence":"default","loops":0,"delay_ms":5000,"ttl_s":300}`.
- `button` "Stop calibration": `payload_press` `{"action":"unlock"}`.
- `binary_sensor` "Calibrating": `state_topic` `xmas_tree/calibrate/state`, `value_template` `{{ 'ON' if value_json.state != 'idle' else 'OFF' }}`.

This is the intended phone flow: press the button in HA, then walk round the tree filming each angle for at least one loop. No laptop is needed.

### 7.3 Light commands while locked

- **Power OFF** MUST unlock immediately and turn the tree off. The user must always be able to turn it off.
- **Brightness and effect changes** MUST be accepted, published in the light state as usual, and applied when the lock ends.
- **Power ON** while locked is a no-op (the tree is already "on" for calibration).

## 8. Default sequence on the device

The firmware bundle MUST include the standard sequence. The canonical copy in the repository is `sequences/default.tcal`, generated by `./calibrate.py sequence --out sequences/default.tcal`. The firmware build (`tools/build_firmware.py`) copies it into the firmware slot as `/fw/<slot>/data/calib_default.tcal`, so it is versioned and rolled back with the firmware. A `/calib_default.tcal` at the flash root is read as a fallback. `play?sequence=default` reads it from flash. If neither file exists, play returns 409 with the error `"no default sequence"`.

When the default sequence changes, only this file changes; the firmware does not.

## 9. Standard sequence layout (informative)

This section describes what the host puts in the file. The firmware does not need to understand it. The parameters are chosen by the benchmark in `docs/calibration-findings.md` and may change without a firmware change.

| Segment | Frames | Content |
|---|---|---|
| lead | 1 × `lead_ms` (1000) | all off. Gives the camera time to settle and marks the loop start. |
| sync | 13 × symbol | Barker-13 pattern of all-on / all-off frames. The decoder correlates whole-image brightness against it to find `T0` and the symbol clock to sub-frame precision. |
| refs | 2 × symbol | all off, then all on. |
| training | 3 × symbol (colour mode) | all LEDs red only, then green only, then blue only. The decoder learns a per-pixel 3×3 colour-mixing matrix from these. |
| code | `bits` × 2 × symbol | for each code bit, the pattern frame then its complement. Each LED shows its codeword bit and then the inverse, so every LED is lit exactly half the time. The decoder compares each pair, which removes exposure drift, ambient light and bloom from neighbours. Colour mode puts three bits in each frame, one per channel. |
| tail | 13 × symbol | Barker-13 again, used to measure clock drift across the loop. |

Levels: all-on frames light all 500 LEDs, so they use `floor(budget / 1500)` (33). Code and training frames use one shared level, the highest that keeps the brightest code frame under the budget (64). Every frame stays under the cap.

## 10. Notes for the browser page (informative)

Phone browsers only allow camera access (`getUserMedia`) on HTTPS or localhost pages. An HTTPS page cannot `fetch()` the tree's plain-HTTP API (mixed content). So the page can drive the tree directly only when it is served over plain HTTP from the LAN, and then it cannot use the camera. The design in `docs/calibration-web.md` therefore triggers playback through Home Assistant or MQTT, or simply lets the tree loop, and relies on the in-video sync preamble rather than a shared clock. The CORS headers above still help laptop tools and LAN pages.

## 11. Changes from the current firmware

- Calibration frames go through the output stage and power cap (was: direct `strip.write()`).
- The lock has a TTL, restores state on exit, and MQTT stays serviced (was: no timeout, MQTT paused).
- New endpoints: `/calibrate` (GET), `/calibrate/sequence`, `/calibrate/play`, `/calibrate/halt`, `/calibrate/frame`, `/calibrate/log`. The legacy endpoints keep working.
- A 100 ms polling loop is no longer used for calibration. Static frames are shown within one frame of the request, and playback uses absolute deadlines.
