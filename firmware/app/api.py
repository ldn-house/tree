"""HTTP routes (port cfg.HTTP_PORT, default 8080). Calibration routes are in
app.calib. "auth" routes need the HTTP_TOKEN (see app.httpd).

    GET  /, /status                 status and diagnostics
    GET  /health                    {"ok": bool, "problems": [...], "warnings": [...]}
    GET  /log                       recent log lines
    GET  /frame                     last frame sent to the strip (base64, wire order)
    GET  /mqtt                      cached MQTT client state
    GET  /effects                   effects, params, overlays, failures
    POST /state               auth  {"power", "brightness", "floor", "colour_accuracy", "effect", "speed",
                                     "auto_s", "params": {effect: {param: v}}}
    GET  /takeover                  active takeovers (app.takeover)
    POST /takeover            auth  start/refresh/end a takeover (JSON, app.takeover)
    DELETE /takeover          auth  end every takeover (?id=X: just that one)
    PUT  /effects/<name>      auth  hot-push one effect module (no reboot); with
                                    UPDATE_KEY set it must be signed (X-Signature)
    DELETE /effects/<name>    auth  remove a hot-pushed effect
    PUT  /files/coords.txt    auth  coordinates in the contract frame
    PUT  /files/coords_compact.txt  auth  legacy triangulate.py output
    PUT  /config              auth  replace config.py from JSON values; signed
                                    (X-Signature) when UPDATE_KEY is set; reboots
    POST /update              auth  firmware bundle (?dry_run=1 to only verify,
                                    ?allow_downgrade=1 to accept an older generation)
    GET  /update/status             progress of the current/last update
    POST /update/github       auth  self-update (?channel=main|stable)
    GET  /update/github/check auth  latest version on a channel
    POST /update/confirm      auth  confirm a trial update now
    POST /update/rollback     auth  boot the previous firmware
    POST /reboot              auth
"""

import gc
import sys

import bootmgr

from app import coords, fx as fxmod, recovery, trace
from app.httpd import HTTPError
from app.log import lines, log

MAX_EFFECT = 32768
MAX_CONFIG = 4096
MAX_COORDS = 65536


def _sha_check(req, data):
    want = req.headers.get("x-content-sha256")
    if want:
        import hashlib

        if bootmgr.hexlify(hashlib.sha256(data).digest()) != want.strip().lower():
            raise HTTPError(400, "X-Content-SHA256 does not match the body")


def register(rt, srv):
    async def h_status(req):
        return 200, rt.status()

    async def h_health(req):
        return 200, {"ok": not rt.sup.problems, "healthy": rt.sup.healthy, "problems": rt.sup.problems,
                     "warnings": [recovery.MESSAGE] if recovery.OUTDATED else []}

    async def h_log(req):
        return 200, {"lines": lines()}

    async def h_frame(req):
        import binascii

        buf = rt.engine.last
        return 200, {"n": rt.n, "order": rt.cfg.COLOR_ORDER, "frame": binascii.b2a_base64(buf).decode().strip()}

    async def h_mqtt(req):
        return 200, rt.mqtt.status() if rt.mqtt else {"configured": False}

    async def h_gc(req):
        gc.collect()
        return 200, {"free_memory": gc.mem_free()}

    async def h_effects(req):
        out = []
        for name in rt.fx.names():
            m = rt.fx.meta(name)
            out.append(
                {
                    "name": name,
                    "duration_ms": m["duration_ms"],
                    "needs_coords": m["needs_coords"],
                    "params": {k: _param_info(v, rt.state.param(name, k, v[0])) for k, v in m["params"].items()},
                    "auto": m.get("auto", True),
                    "overlay": rt.fx.overlay_file(name) is not None,
                    "eligible": rt.engine.eligible(name),
                    "error": rt.engine.errors.get(name),
                }
            )
        return 200, {"effects": out, "running": rt.engine.name, "fx_error": rt.fx.error}

    async def h_state(req):
        body = await req.json()
        changes = {}
        for k in ("power", "brightness", "floor", "colour_accuracy", "effect", "speed", "auto_s", "params",
                  "transition_style", "transition_s", "transition"):
            if k in body:
                changes[k] = body[k]
        try:
            changed = rt.apply(changes, "http")
        except ValueError as e:
            raise HTTPError(400, str(e))
        return 200, {"changed": changed, "state": rt.state.snapshot()}

    async def h_takeover(req):
        return 200, rt.takeovers.status()

    async def h_takeover_post(req):
        try:
            res = rt.takeovers.command(await req.json(), "http")
        except ValueError as e:
            raise HTTPError(400, str(e))
        return 200, {"result": res, "takeover": rt.takeovers.status()}

    async def h_takeover_delete(req):
        tid = req.query.get("id")
        cmd = {"action": "end", "id": tid} if tid else {"action": "end_all"}
        try:
            res = rt.takeovers.command(cmd, "http")
        except ValueError as e:
            raise HTTPError(400, str(e))
        return 200, {"result": res, "takeover": rt.takeovers.status()}

    async def h_effect_put(req):
        name = req.path[len("/effects/") :]
        if not fxmod.valid_name(name):
            raise HTTPError(400, "effect names are lowercase letters, digits and _")
        data = await req.body(MAX_EFFECT)
        _sha_check(req, data)
        # Effect code runs with full access to the flash (there is no sandbox
        # in MicroPython), so with a signing key the token alone is not
        # enough: check the signature before any of it is executed (H4).
        if not bootmgr.signature_ok("effect", name.encode() + b"\n" + data, req.headers.get("x-signature")):
            raise HTTPError(403, "effects must be signed with UPDATE_KEY (X-Signature header)")
        is_mpy = req.query.get("format") == "mpy" or (len(data) > 4 and data[0] == 0x4D and data[1] < 32)
        ext = ".mpy" if is_mpy else ".py"
        try:
            if is_mpy:
                bootmgr.check_mpy(data[:4])
            else:
                bootmgr.check_source(name + ".py", data.decode())
        except (bootmgr.InstallError, UnicodeError) as e:
            raise HTTPError(400, str(e))
        result = _try_effect(rt, name, data, ext)
        rt.fx.forget(name)
        rt.engine.reload_effect(name)
        activate = req.query.get("activate") in ("1", "true", "yes")
        if activate:
            rt.apply({"effect": name}, "http")
        if rt.ha:
            rt.ha.rediscover_at = 0
        log("fx: hot-pushed %s (%d bytes)" % (name, len(data)))
        result.update({"effect": name, "overlay": True, "activated": activate})
        return 200, result

    async def h_effect_delete(req):
        name = req.path[len("/effects/") :]
        removed = False
        for ext in (".py", ".mpy"):
            p = "%s/%s%s%s" % (fxmod.OVERLAY, fxmod.PREFIX, name, ext)
            if bootmgr.exists(p):
                import os

                os.remove(p)
                removed = True
        if not removed:
            raise HTTPError(404, "no hot-pushed effect named " + name)
        rt.fx.forget(name)
        rt.engine.reload_effect(name)
        if rt.state.get("effect") == name and name not in rt.fx.names():
            rt.apply({"effect": "auto"}, "http")
        if rt.ha:
            rt.ha.rediscover_at = 0
        return 200, {"effect": name, "removed": True, "packaged": name in rt.fx.names()}

    async def h_coords(req):
        legacy = req.path.endswith("coords_compact.txt")
        return await _store_coords(req, legacy)

    async def _store_coords(req, legacy):
        data = await req.body(MAX_COORDS)
        _sha_check(req, data)
        try:
            from effects import _coords

            parsed = _coords.parse(data.decode())
        except Exception as e:
            raise HTTPError(400, "cannot parse coordinates: %r" % (e,))
        if len(parsed) > rt.n:
            raise HTTPError(400, "%d coordinates for %d LEDs" % (len(parsed), rt.n))
        target = "coords_compact.txt" if legacy else coords.USER_FILE
        bootmgr.write_atomic(target, data)
        if legacy and bootmgr.exists(coords.USER_FILE):
            log("coords: legacy file stored but coords.txt takes precedence")
        rt.engine.set_coords(coords.load(rt.n, rt.slot_dir))
        return 200, {"stored": target, "lines": len(parsed), "known": coords.count(rt.engine.coords), "leds": rt.n}

    async def h_update(req):
        if "file" in req.query:
            f = req.query["file"]
            if f in ("coords_compact.txt", "coords.txt"):
                return await _store_coords(req, f == "coords_compact.txt")
            raise HTTPError(400, "single-file updates are gone: push a firmware bundle (./flash.py --ota)")
        if req.length is None:
            raise HTTPError(411, "Content-Length required")
        dry = req.query.get("dry_run") in ("1", "true", "yes")
        downgrade = req.query.get("allow_downgrade") in ("1", "true", "yes")
        try:
            m = await rt.updates.install_stream(req.read_chunk, req.length, "http", dry_run=dry,
                                                allow_downgrade=downgrade)
        except HTTPError:
            raise
        except bootmgr.InstallError as e:
            raise HTTPError(409 if "in progress" in str(e) or "not confirmed" in str(e) else 400, str(e))
        if dry:
            return 200, {"verified": m.get("version"), "files": len(m["files"])}
        rt.request_reboot("installed update over HTTP", 800)
        return 200, {"installed": m.get("version"), "slot": bootmgr.read_state()["active"],
                     "rebooting": True, "trial_s": rt.cfg.TRIAL_HEALTHY_S}

    async def h_config(req):
        body = await req.body(MAX_CONFIG)
        if not bootmgr.signature_ok("config", body, req.headers.get("x-signature")):
            raise HTTPError(403, "config uploads must be signed with UPDATE_KEY (X-Signature header)")
        try:
            import json

            keys = bootmgr.install_config(json.loads(body))
        except (ValueError, SyntaxError) as e:
            raise HTTPError(400, str(e))
        rt.request_reboot("config.py replaced over HTTP (trial)")
        return 200, {"written": keys, "rebooting": True, "trial_s": rt.cfg.TRIAL_DEADLINE_S}

    async def h_update_status(req):
        return 200, rt.updates.status()

    async def h_github(req):
        err = rt.updates.start_github(req.query.get("channel", rt.cfg.UPDATE_CHANNEL))
        if err:
            raise HTTPError(409, err)
        return 202, {"started": True, "status": "/update/status"}

    async def h_github_check(req):
        if not rt.wifi.up:
            raise HTTPError(409, "wifi is down")
        return 200, await rt.updates.check(req.query.get("channel"))

    async def h_confirm(req):
        confirmed = bootmgr.mark_healthy(rt.slot)
        rt.info["trial"] = False
        rt.sup.confirmed = True
        return 200, {"confirmed": confirmed}

    async def h_rollback(req):
        slot = bootmgr.rollback()
        if slot is None:
            raise HTTPError(409, "no previous firmware to roll back to")
        rt.request_reboot("rollback requested over HTTP")
        return 200, {"rebooting": True, "slot": slot}

    async def h_reboot(req):
        rt.request_reboot("reboot requested over HTTP")
        return 200, {"success": True, "message": "Rebooting..."}

    srv.route("GET", "/", h_status)
    srv.route("GET", "/status", h_status)
    srv.route("GET", "/health", h_health)
    srv.route("GET", "/log", h_log)
    srv.route("GET", "/frame", h_frame)
    srv.route("GET", "/mqtt", h_mqtt)
    srv.route("GET", "/gc", h_gc)
    srv.route("GET", "/effects", h_effects)
    srv.route("POST", "/state", h_state, auth=True)
    srv.route("GET", "/takeover", h_takeover)
    srv.route("POST", "/takeover", h_takeover_post, auth=True)
    srv.route("DELETE", "/takeover", h_takeover_delete, auth=True)
    srv.route("PUT", "/effects/*", h_effect_put, auth=True, deadline=60)
    srv.route("POST", "/effects/*", h_effect_put, auth=True, deadline=60)
    srv.route("PUT", "/config", h_config, auth=True)
    srv.route("POST", "/config", h_config, auth=True)
    srv.route("DELETE", "/effects/*", h_effect_delete, auth=True)
    for p in ("/files/coords.txt", "/files/coords_compact.txt"):
        srv.route("PUT", p, h_coords, auth=True, deadline=60)
        srv.route("POST", p, h_coords, auth=True, deadline=60)
    srv.route("POST", "/update", h_update, auth=True, deadline=300)
    srv.route("GET", "/update/status", h_update_status)
    srv.route("GET", "/update/github/status", h_update_status)
    srv.route("POST", "/update/github", h_github, auth=True)
    srv.route("GET", "/update/github/check", h_github_check, auth=True, deadline=60)
    srv.route("POST", "/update/confirm", h_confirm, auth=True)
    srv.route("POST", "/update/rollback", h_rollback, auth=True)
    srv.route("POST", "/reboot", h_reboot, auth=True)


def _param_info(spec, value):
    info = {"default": spec[0], "min": spec[1], "max": spec[2], "value": value}
    if fxmod.is_rgb(spec):
        info["type"] = "rgb"
    return info


def _try_effect(rt, name, data, ext):
    """Write a pushed effect to a staging module, import it, build it and
    render two frames. Only if that all works does it replace the overlay."""
    import os

    stage_mod = fxmod.PREFIX + name + fxmod.STAGING
    stage = "%s/%s%s" % (fxmod.OVERLAY, stage_mod, ext)
    bootmgr.makedirs(fxmod.OVERLAY)
    with open(stage, "wb") as f:
        f.write(data)
    trace.at(trace.EFFECT_START)  # its code runs here, inside the request
    try:
        if stage_mod in sys.modules:
            del sys.modules[stage_mod]
        try:
            mod = __import__(stage_mod)
            if not hasattr(mod, "Effect"):
                raise ValueError("module has no Effect class")
            meta = fxmod.module_meta(mod)
            from app.engine import _ctx

            params = {k: v[0] for k, v in meta["params"].items()}
            inst = mod.Effect(_ctx(rt.n, rt.engine.coords, params))
            frame = bytearray(3 * rt.n)
            inst.render(0, frame)
            inst.render(33, frame)
        except Exception as e:
            raise HTTPError(400, "effect failed to load or render: %r" % (e,))
        finally:
            if stage_mod in sys.modules:
                del sys.modules[stage_mod]
            gc.collect()
        for other in (".py", ".mpy"):
            p = "%s/%s%s%s" % (fxmod.OVERLAY, fxmod.PREFIX, name, other)
            if other != ext and bootmgr.exists(p):
                os.remove(p)
        os.rename(stage, "%s/%s%s%s" % (fxmod.OVERLAY, fxmod.PREFIX, name, ext))
    finally:
        if bootmgr.exists(stage):
            os.remove(stage)
    return {"duration_ms": meta["duration_ms"], "params": meta["params"], "needs_coords": meta["needs_coords"],
            "auto": meta["auto"]}
