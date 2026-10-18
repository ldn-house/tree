# How to calibrate the tree

Calibrating teaches the tree where each of its 500 lights is in 3D, so effects like a sweeping plane or a spiral can follow the shape of the tree. It takes about ten minutes, one phone, and a dark room.

![Explainer preview](img/calibration/calibrate-the-tree.gif)

▶ The full 70-second explainer video (no sound, everything is captioned): [calibrate-the-tree.mp4](img/calibration/calibrate-the-tree.mp4).

## What the button does

In Home Assistant, on the Christmas tree's page, press **Play calibration sequence**. Then:

1. **The tree goes dark for about 6 seconds.**
2. **It starts blinking in a fast, flickery pattern**, which repeats every 6.6 seconds. Each round starts with a second of darkness and a burst of quick whole-tree flashes.
3. **It keeps repeating for up to 5 minutes** after you pressed the button.
4. **Then it goes back to whatever it was showing before.**

While it's running, the tree's **Calibrating** sensor (in the Diagnostic section of its page) says **On**. If the tree doesn't go dark, or Calibrating stays Off, stop and tell whoever looks after the tree.

Pressing the button again while it's still blinking does nothing. If the 5 minutes run out before you're done, wait for the tree to go back to normal, then press it again. **Stop calibration** ends it early.

## Before you start

- **Use one phone for every clip.** Don't swap phones halfway.
- **Make the room dark.** Room lights off, curtains closed, lamps and screens off or out of shot. Darker is better, but you need just enough light to move safely.
- **Use normal video at 1080p**, at 30 or 60 fps. On an iPhone, tap the "4K" or "HD" label at the top of the camera screen until it says HD. Don't use Cinematic, Slo-mo, Time-lapse, Action or Night mode.
- **iPhone only: turn off HDR video.** It's in Settings → Camera → Record Video → HDR Video.
- **Hold the phone upright** (portrait), so the whole tree fits.
- **Pick the "front"** of the tree: the side people usually look at. The first clip must show the front; effects treat that side as the front.

## Step by step

There are two ways to get the different angles. Both work equally well; pick whichever suits the room.

**Walk around the tree** if there's space all the way round it:

1. **Choose 4 to 6 spots** spread evenly around the tree, about 2 metres away. Start with the front, then go round in order. More spots help the lights at the back.
2. At each spot, **frame the whole tree**, from the top to the bottom branches, with a little space around it.

**Or turn the tree on the spot** if it's against a wall:

1. **Put the phone in one place** about 2 metres away, on a tripod or propped on something steady, framing the whole tree. It stays there for every clip.
2. **Film the front first**, then **turn the tree by about 60–90°** between clips, always the same way round, until it has gone all the way round: 4 to 6 clips.
3. **After each turn, wait a few seconds for the branches to stop swaying** before you record.
4. It's fine that the room behind the tree looks the same in every clip; only the lights matter.

**Then, either way:**

1. **Press Play calibration sequence.** Wait for the tree to go dark and then start blinking.
2. **Start recording and hold still for about 12 seconds** (count slowly to 12). Lean on a wall or a chair back, or use a tripod. Small wobbles are fine; walking isn't.
3. **Stop recording**, move to the next spot or turn the tree, and record again. The tree keeps blinking, so there's no need to press the button again.
4. When you've done every angle, you're finished. The tree returns to normal by itself, or press **Stop calibration**.

Six angles take about 3 minutes, well inside the 5-minute window.

**Two clips from the same angle don't help.** If you film the same spot twice (for example because the first clip was too short), that's fine, but it doesn't count as an extra angle. The computer side will warn about it.

## Adding angles later

If the result has gaps (lights that only one clip could see), you can add angles another evening without redoing everything, as long as **the lights haven't been moved or re-hung** and you **use the same phone on the same 1× lens**. Turning the whole tree is fine; moving individual branches or the wire isn't.

Film just the new angles, send them over, and the computer side decodes only the new clips and solves again with the old and new ones together.

## Please don't

- **Zoom, or switch lenses** (0.5×, 2× and so on). Stay on the normal 1× lens for every clip.
- **Move while recording**: walking, or turning the tree. Stop, move or turn, let the branches settle, then record.
- **Stand so close** that the top or bottom of the tree is cut off.
- **Film with the room lights on**, or with the phone's torch on.
- **Start recording before the tree starts blinking.** Press the button first, wait for the blinking, then record.

## Send the clips

**AirDrop the clips to the Mac** of whoever runs the computer side, or copy them over a cable. **Don't send them through WhatsApp, Messages or email**: those shrink the video, and the decoder needs the originals.

It helps to name them in the order you filmed, front first: `1-front`, `2`, `3`, and so on. When adding angles later, carry on numbering from where you stopped.

> **Why does this work?**
>
> Every light blinks its own number, as a pattern of on and off, followed by the exact opposite. Comparing each pair tells the decoder "on" from "off" even when the camera's brightness changes. The pattern includes error correction, so a light that's misread is rejected rather than mistaken for another light.
>
> In each clip, the decoder works out which light is where on the screen. A single view only gives a direction from the phone to each light. Several views from different spots cross at each light, and that pins it down in 3D. The decoder also works out where each phone spot was, and the camera's lens, from the lights themselves.

## For whoever runs the computer bit

This needs the tree repository on branch `calibration/video-pipeline` (or `main` once #22 is merged), with [uv](https://docs.astral.sh/uv/) installed. Put the clips in a folder, for example `~/Desktop/tree-clips`, named so the front clip sorts first.

**Before anyone films, check the tree has the calibration sequence.** If it doesn't, the button does nothing.

```bash
curl -s http://tree.local:8080/calibrate
```

`"default"` under `"sequences"` must not be `null`; with the current sequence it shows `"crc32": "83cffe28"`. If it is `null`, the firmware needs a build from #21 that bundles `sequences/default.tcal`, which is a firmware update.

```bash
# 1. Find the LEDs in each clip. Clips are decoded in parallel, with progress per clip;
#    five 1080p 60 fps clips take about 30 seconds on a recent Mac.
./calibrate.py decode-video ~/Desktop/tree-clips/* --out positions/tonight

# 2. Combine the views into 3D (front clip first, which the name order handles)
./calibrate.py solve positions/tonight/*_detections.json --out coordinates

# 3. Optional: check for wire-continuity problems (assumes a 1.75 m light span and 10 cm spacing)
./calibrate.py quality coordinates/coords.txt --height-m 1.75 --pitch 0.10

# 4. Install it (needs the tree's HTTP token, as for flash.py)
curl -X PUT --data-binary @coordinates/coords.txt \
  -H "Authorization: Bearer $TREE_TOKEN" http://tree.local:8080/files/coords.txt
```

**Adding angles later:** decode only the new clips into the same folder (`./calibrate.py decode-video ~/Desktop/tree-clips/6.mov --out positions/tonight`), then run step 2 again with all of them; the front clip still has to sort first.

**On a slow or low-memory computer**, `--max-fps 30` skips every other frame of a 60 fps clip, and `--scale 0.5` decodes at half size (a quarter of the memory). Both are slightly less thorough: on the first real calibration, `--scale 0.5` found 12–18% fewer lights.

**What to look for.**

- **Step 1** prints "N/500 confident detections" per clip. A few hundred per clip is normal; lights behind the tree can't be seen from every side.
- **Step 2** prints a median reprojection error. About 1 px or less means the views agree. Much more than about 8 px means something went wrong, usually a zoomed or moving clip. Leave that clip out and run it again.
- **Step 2 warns** "… look like the same viewpoint" when two clips were filmed from the same angle. Lights seen only in those two clips are treated as seen once. If a clip can't be placed at all, it's left out with a warning, and the rest are solved.
- **Step 2** also fills in LEDs seen in fewer than two angles along the wire, and marks them with low confidence in `coordinates/coords_metres.csv`.
- **Step 4** takes effect straight away; `coords.txt` takes precedence over any old `coords_compact.txt`.

**First real calibration (18 October):** five clips, filmed by turning the tree in front of a fixed phone; two were from the same angle. 453 of 500 lights were seen at least once, the views agreed to 0.65 px, and the shape came out as a clean cone. If a run looks off, keep the clips and the `positions/tonight` folder so it can be checked.
