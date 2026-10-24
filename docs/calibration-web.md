# Calibration web page and other ideas (design only)

Status: design, not built. It builds on [the protocol](calibration-protocol.md) and the decoder in `calib/video.py`; the numbers quoted are from [the findings report](calibration-findings.md).

## 1. Goal

A person with a phone and no laptop should be able to recalibrate the tree in about five minutes: start the sequence, film the tree from four or more spots, and get a new `coords.txt` installed. A laptop-based flow (`calibrate.py decode-video` on files copied off the phone) stays the reference implementation and the fallback.

## 2. The browser constraints that shape the design

| Constraint | Consequence |
|---|---|
| `getUserMedia` (camera) needs a secure context: HTTPS or `localhost`. | The page cannot be served by the tree itself (plain HTTP on the LAN) if it wants the camera. |
| An HTTPS page cannot `fetch()` a plain-HTTP LAN address (mixed content), and Chrome's Private Network Access adds a preflight on top. | An HTTPS page cannot call the tree's HTTP API directly, CORS headers or not. |
| Phone browsers throttle or kill background tabs and limit memory (about 1–2 GB on iOS Safari). | Decoding has to stream frames, not hold a whole clip of full-resolution frames. |
| `MediaRecorder` output (WebM/MP4) is hard to seek frame-accurately in the browser; `requestVideoFrameCallback` gives per-frame presentation and capture times on live streams. | Process frames live as they arrive rather than record-then-decode, or record and decode on a laptop. |

The protocol is designed so that none of this matters for timing. The tree loops the sequence, and the decoder finds sync from the Barker preamble in the video itself, so the page never needs to talk to the tree at the moment of capture.

## 3. Recommended architecture

```
 Home Assistant (HTTPS, e.g. Nabu Casa)          phone browser (HTTPS page)
 ┌──────────────────────────────────┐          ┌──────────────────────────────────────┐
 │ button "Play calibration sequence" ◀────────┤ 1. "Start" calls HA REST/WebSocket     │
 │   -> MQTT xmas_tree/calibrate/set │          │    (long-lived token), or the user    │
 └──────────────┬───────────────────┘          │    presses the HA button themselves   │
                │ MQTT                          │ 2. getUserMedia rear camera, 1080p30, │
 ┌──────────────▼───────────────────┐          │    exposure/WB lock if supported      │
 │ tree: lock, loop default.tcal     │  light   │ 3. per frame: sparkle series -> sync, │
 │ for ttl_s (300 s)                 │ ───────▶ │    row-wise symbol accumulation       │
 └──────────────────────────────────┘          │ 4. per view: decode -> detections     │
                                               │ 5. after N views: solve -> coords.txt │
                                               │ 6. upload via HA (file/notify) or     │
                                               │    download for flash.py              │
                                               └──────────────────────────────────────┘
```

- **Hosting.** A static page on GitHub Pages (HTTPS), or served as a Home Assistant panel or add-on ingress page (HTTPS through HA). No server-side code.
- **Triggering.** The page calls HA's REST API (`POST /api/services/button/press`) with a long-lived access token the user pastes once (stored in `localStorage`). Or it does nothing at all, and the user presses the HA button or sends the MQTT command. Since the tree loops for `ttl_s`, the page only has to know the sequence parameters (it fetches `sequences/default.tcal` from the same static site and parses it with a JavaScript port of `seqfile.py`, which is a few dozen lines).
- **Installing the result.** Not from the browser directly (mixed content again). Options in order of preference:
  1. a small HA script that takes the coordinates text and forwards it to the tree's `POST /update?file=coords.txt` from inside the LAN, using the tree's HTTP token stored in HA;
  2. download `coords.txt` and run `./flash.py --ota --host … coords.txt`;
  3. a LAN-only HTTP page served by the tree (no camera) that accepts a pasted or uploaded file.

## 4. Decoding in the browser

Two options, with the same algorithm as `calib/video.py`.

**A. Pyodide (recommended first).** The repo already runs Python in the browser: the 3D effects viewer (`viewer/`, served by `serve.py`) runs the effects in Pyodide. Pyodide ships numpy, scipy and OpenCV, so `calib/video.py` and `calib/solve.py` run unchanged. Frames come from a `<video>` element fed by the camera stream, drawn into an `OffscreenCanvas` at reduced resolution and handed to Python as `Uint8ClampedArray` views (zero copy with `pyodide.ffi.to_py` on a `memoryview`).

- **Pass 1 (sync)** is cheap: a 270 × 480 sparkle series per frame.
- **Pass 2** is needed because decoding needs the phase. Hold the clip for one loop (about 6.6 s) at 540 × 960 luminance in a ring buffer: 200 frames × 0.5 MB = 100 MB, which fits even on iOS. Then accumulate symbols row-wise. Accumulating a second loop on the fly after sync is found avoids the ring buffer entirely: the sequence loops, so the page can sync on loop 1 and accumulate from loop 2 onwards.
- **Cost.** The desktop decoder uses about 5 s of CPU per 540 × 960 clip of 1.6 loops (measured on an M-series Mac, single-threaded numpy and OpenCV). Pyodide is roughly 2–4× slower than native numpy, so 10–20 s per view in a laptop browser and perhaps 30–60 s on a phone. That's tolerable if it runs while the user walks to the next spot; WebGL (option B) is the way to make it instant.
- **Resolution.** At 540 × 960 the benchmark loses little against 1080 × 1920 (see findings), so decoding at half resolution in the browser is not a big compromise.

**B. WebGL / WebGPU.** The per-pixel work (accumulating symbols, soft bits, and the 14 × 500 correlation for each lit pixel) maps directly onto fragment or compute shaders and would run in real time at full resolution. That's worth doing only if Pyodide proves too slow on phones. Blobs and the solve stay on the CPU.

**Live feedback.** Two things make the capture far more reliable for non-experts:

- **A live coverage overlay.** After each view, LEDs decoded in at least two views are drawn as green dots on the camera preview, and LEDs not yet seen in two views as red ghost points projected from the current partial solution. The user can see which side of the tree still needs a shot.
- **A stillness meter.** The decoder assumes a still camera for at least one loop, so the page shows a "hold still" progress ring using frame-difference energy outside the tree, and rejects loops with motion.

**Camera controls.** Where supported (Chrome on Android exposes `exposureMode`, `whiteBalanceMode` and `focusMode` via `MediaStreamTrack.applyConstraints`), the page locks exposure, white balance and focus after the first second. iOS Safari does not expose these. The decoder's exposure-invariant sync and per-pair differencing were designed for that case, and the benchmark shows auto exposure costs little.

## 5. Flow, screen by screen

1. **Connect.** Paste the HA URL and token (once). The page checks `GET /api/states/binary_sensor.xmas_tree_calibrating`.
2. **Start.** "Start calibration" presses the HA button. The tree starts looping at once, each loop opening with 1 s of dark. The page shows the loop length (6.6 s) and "Stand about 2 m from the tree, fit the whole tree in the frame, hold still".
3. **Capture view 1 to N.** For each spot the page waits for stillness, records 1.5 loops, decodes, and shows coverage. It recommends 4–6 spots spread around the tree. More views help occluded LEDs most.
4. **Solve.** BA over all views, with the wire prior. It shows a 3D preview (three.js) with LEDs coloured by confidence, the median reprojection error, and how many LEDs were interpolated.
5. **Verify (optional, recommended).** The page asks the tree, through HA, to play a "plane sweep" effect using the *new* coordinates, and films it. A correct calibration shows a flat plane moving up the tree. The decoder can check this automatically by comparing the lit-LED set per frame with the plane prediction. This catches mirrored or misindexed results before they're installed.
6. **Install.** Through the HA script, or download.

## 6. Other ideas, with feasibility

| Idea | What it gives | Feasibility | Notes |
|---|---|---|---|
| **Walk-around single clip.** The tree loops; the user films one continuous clip, pausing about 2 loops at each of 5–8 spots. The decoder segments still windows automatically. | One recording instead of four; more views for free | **High.** The sync already finds every loop; a stillness detector (background frame difference) picks the usable loops; each becomes a view. | Easiest big UX win. A day of work on top of `calib/video.py`. |
| **Verification by playback** (section 5, step 5) | Catches index, mirror and scale mistakes before install | **High.** Needs a "show the plane at height h" calibration frame, which the static `/calibrate/frame` endpoint already allows. | Recommend making it part of every calibration. |
| **Fixed camera, rotate the tree** (the tue1–4 captures appear to have been done this way) | One tripod position, the tree is turned between loops | **High** for geometry (rigid motion is equivalent to moving the camera). | The solver doesn't care. Turning the tree may shake branches, so wait a moment before each loop. |
| **Continuous moving-camera decoding** (no stops, slow orbit) | Fastest capture | **Low to medium.** The per-pixel decoding assumes a static image for the whole code segment (about 4 s). Making it work needs either frame-to-frame registration (homographies are wrong for a 3D tree) or per-LED tracking of blinking blobs. Point-tracking across 30 frames of blinking LEDs is plausible but error-prone. | Not worth it while stop-and-hold is simple. |
| **Two phones at once** | Halves capture time | **High.** Nothing in the protocol is single-camera. | Pure UX. |
| **Shorter code: 9 data bits + 1 parity, colour multiplexed** | About 1 s of code frames | **Medium.** Works in simulation (findings), but colour depends on the camera's crosstalk and white balance, so it needs the real-camera test in the findings report first. | Keep mono Hamming as the default. |
| **Self-check for wire-order errors.** Some strings are wired in reverse, or have a dead pixel that shifts indices. A short pattern lights the first and last 10 LEDs in distinct colours. | Detects a reversed string or missing pixel | **High.** A two-frame `.tcal`. | Cheap insurance; add to the web flow's first screen. |
| **Use reflections as extra views** (window glass, a mirror behind the tree) | More views of occluded LEDs | **Low.** Needs mirror-plane estimation and handedness flip in BA. | Research toy. |
| **Phone depth (LiDAR / ToF)** | Metric scale directly | **Low in the browser** (no depth API). **Medium in a native app** (ARKit `sceneDepth`). | The wire pitch or a measured tree height already gives scale well enough. |
| **Photometric LED orientation** | Brightness per view gives each bullet LED's pointing direction | **Medium.** The data is already there (amplitude per view). | Could make effects "sparkle towards the viewer"; a fun follow-up. |
| **HA camera recheck.** A fixed camera (HA camera entity) films one sequence loop every night. | Detects drift (a moved tree, re-hung string) | **Medium.** A one-view decode can't re-triangulate, but it can compare 2D projections with the stored solution's reprojection and flag LEDs that moved more than a few pixels. | Needs a stored camera pose for that camera, which BA can provide if that camera is included once in a full calibration. |
