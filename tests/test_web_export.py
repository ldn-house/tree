"""Exercise the published format, actual renderer, and archive/preview CLI."""

import hashlib
import json
import re
import subprocess
import sys
import tarfile

import numpy as np
import pytest
from PIL import Image

import effects
import export_web
import render


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("web-export")
    out = root / "first"
    archive = root / "tree-effects.tar.gz"
    subprocess.run([sys.executable, str(render.ROOT / "export_web.py"),
                    "--out", str(out), "--archive", str(archive)], check=True)
    return root, out, archive, json.loads((out / "manifest.json").read_text())


def decode(out, entry, leds):
    return np.frombuffer((out / entry["file"]).read_bytes(), np.uint8).reshape(entry["frames"], leds, 3)


def test_manifest_and_binaries(exported):
    _, out, _, manifest = exported
    assert manifest["format"] == 1
    assert re.fullmatch(r"[0-9a-f]{40}", manifest["source"]["commit"])
    assert manifest["fps"] == 20
    assert manifest["leds"] == 500
    assert manifest["seed"] == 0
    assert manifest["encoding"] == "rgb8"
    assert manifest["frame_stage"] == "raw"
    assert manifest["output"] == {"applied": False, "brightness": 255, "limit_a": 1.5,
                                   "ma_per_channel": 8, "gamma": None, "brightness_floor_pct": 4}
    coords = export_web.calibrated_coords()
    assert manifest["coords"] == [None if c is None else [round(v, 4) for v in c] for c in coords]
    assert len(manifest["coords"]) == manifest["leds"]
    assert [e["name"] for e in manifest["effects"]] == effects.EFFECTS
    assert not set(effects.DISABLED) & {e["name"] for e in manifest["effects"]}
    for e in manifest["effects"]:
        assert re.fullmatch(r"[a-z0-9_]+\.bin", e["file"])
        assert e["label"]
        data = (out / e["file"]).read_bytes()
        assert len(data) == e["byte_length"] == e["frames"] * manifest["leds"] * 3
        assert hashlib.sha256(data).hexdigest() == e["sha256"]
        assert len(export_web.gzip_bytes(data)) == e["gzip_bytes"] <= 300_000
        frames = decode(out, e, manifest["leds"])
        assert frames.max() > 0


def test_deterministic_in_separate_processes_and_archive(exported):
    root, out, archive, manifest = exported
    second, second_archive = root / "second", root / "second.tar.gz"
    subprocess.run([sys.executable, str(render.ROOT / "export_web.py"),
                    "--out", str(second), "--archive", str(second_archive)], check=True)
    names = ["manifest.json"] + [e["file"] for e in manifest["effects"]]
    assert sorted(p.name for p in out.iterdir()) == sorted(names)
    for name in names:
        assert (out / name).read_bytes() == (second / name).read_bytes()
    assert archive.read_bytes() == second_archive.read_bytes()
    assert archive.read_bytes()[4:8] == bytes(4)  # gzip timestamp
    with tarfile.open(archive) as tar:
        assert tar.getnames() == names
        for member in tar.getmembers():
            assert member.isfile() and member.mtime == 0
            assert member.uid == member.gid == 0
            assert member.uname == member.gname == ""
            assert tar.extractfile(member).read() == (out / member.name).read_bytes()
    # A stale effect/preview in the directory must not enter the release.
    (second / "unregistered.bin").write_bytes(b"stale")
    export_web.write_archive(second, manifest, second_archive)
    assert second_archive.read_bytes() == archive.read_bytes()


@pytest.mark.parametrize("name", effects.EFFECTS)
def test_renderer_parity_and_loop_seam(exported, name):
    _, out, _, manifest = exported
    e = next(e for e in manifest["effects"] if e["name"] == name)
    loop = e["loop"]
    count, fade = e["frames"], loop["crossfade_frames"]
    reference_count = count + fade if fade else count + 1
    reference = render.run_effect(name, seconds=reference_count / manifest["fps"],
                                  fps=manifest["fps"], seed=manifest["seed"],
                                  brightness=manifest["output"]["brightness"] / 255,
                                  coords=export_web.calibrated_coords(), keep_raw=True)["raw"]
    actual = decode(out, e, manifest["leds"])
    start = loop["source_start_frame"]
    # Every untouched frame, including random/stateful effects, is byte exact.
    np.testing.assert_array_equal(actual[:count - fade], reference[start:start + count - fade])
    if fade:
        for i in range(fade):
            expected = (reference[count + i].astype(np.int32) * (fade - i - 1)
                        + reference[i].astype(np.int32) * (i + 1)) // fade
            np.testing.assert_array_equal(actual[count - fade + i], expected)
        # The wrap is exactly one real source-frame step, not a frozen head.
        np.testing.assert_array_equal(actual[-1], reference[fade - 1])
        np.testing.assert_array_equal(actual[0], reference[fade])
    else:
        # Natural loops exclude the duplicate endpoint, retaining uniform time.
        np.testing.assert_array_equal(actual[0], reference[count])


def test_render_cli_parity(exported):
    root, out, _, manifest = exported
    e = manifest["effects"][0]
    destination = root / "renderer-cli"
    subprocess.run([sys.executable, str(render.ROOT / "render.py"), e["name"],
                    "--seconds", str(e["frames"] / manifest["fps"]),
                    "--fps", str(manifest["fps"]), "--seed", str(manifest["seed"]),
                    "--brightness", "1", "--view", "raw", "--no-anim", "--sheet-frames", "1",
                    "--out", str(destination)], check=True)
    metrics = json.loads((destination / e["name"] / "metrics.json").read_text())
    assert metrics["exceptions"] == []
    assert metrics["view"] == "raw"
    result = render.run_effect(e["name"], seconds=e["frames"] / manifest["fps"],
                               fps=manifest["fps"], seed=manifest["seed"],
                               coords=export_web.calibrated_coords(), keep_raw=True)
    assert hashlib.sha256(result["raw"].tobytes()).hexdigest() == e["sha256"]
    # render.py's metrics hash describes wire output even with --view raw.
    # Compare the CLI sheet pixels using the actual exported binary frames.
    result["frames"] = decode(out, e, manifest["leds"])
    views = render.Views(render.display_positions(export_web.calibrated_coords())[0])
    expected = render.contact_sheet(result, views, cells=1)
    with Image.open(destination / e["name"] / "sheet.png") as image:
        np.testing.assert_array_equal(np.asarray(image), np.asarray(expected))


def test_preview_reads_binaries_without_running_effects(exported, monkeypatch):
    root, out, _, _ = exported

    def forbidden(*args, **kwargs):
        pytest.fail("preview must not load or render effects")

    monkeypatch.setattr(effects, "load", forbidden)
    monkeypatch.setattr(render, "run_effect", forbidden)
    path = root / "decoded.png"
    export_web.preview(out, path)
    with Image.open(path) as image:
        assert image.width == 960
        assert image.height > 2000
        assert np.asarray(image).std() > 10


def test_missing_invalid_and_unplaced_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "coords.txt"
    monkeypatch.setattr(export_web, "COORDS_PATH", path)
    with pytest.raises(FileNotFoundError):
        export_web.calibrated_coords()
    path.write_text("0,0,0\n")
    with pytest.raises(ValueError):
        export_web.calibrated_coords()
    path.write_text("nan,0,0\n" * 500)
    with pytest.raises(ValueError):
        export_web.calibrated_coords()
    path.write_text("0,0,0\n" + "null\n" * 499)
    coords = export_web.calibrated_coords()
    assert len(coords) == 500 and coords[1:] == [None] * 499


def test_partial_render_fails(monkeypatch):
    monkeypatch.setattr(render, "run_effect", lambda *a, **kw: {
        "frames": [], "metrics": {"exceptions": [{"error": "synthetic render failure"}]},
    })
    with pytest.raises(RuntimeError, match="incomplete render"):
        export_web.render_frames("rainbow_cycle", 1, None)
