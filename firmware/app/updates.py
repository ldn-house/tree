"""Firmware updates: bundle uploads and self-update from GitHub Releases.

Both paths stream a bundle into bootmgr.Installer, which writes the inactive
slot and verifies size, SHA-256, signature, .mpy compatibility and Python
syntax before anything is switched. Only then is the slot committed (one
atomic boot.json rename) and the board rebooted into it as a trial. If the
new firmware does not report healthy, bootmgr rolls back on the next boot.

GitHub Releases replace nightly.link artifacts (which expire, review H6). CI
publishes tree-firmware.bin and manifest.json for two channels:
    stable: the latest vX.Y.Z release  (releases/latest/download/...)
    main:   rolling "firmware-main" prerelease, rebuilt on every push to main
Only signed bundles are accepted from GitHub: TLS on the device is not
certificate-checked (no CA store or clock), so integrity comes from the
HMAC signature made by CI with the UPDATE_KEY secret.
"""

import json
import time

import asyncio

import bootmgr

from app import dns
from app.log import log

CHANNELS = ("main", "stable")
ASSET = "tree-firmware.bin"
MANIFEST = "manifest.json"
TIMEOUT = 15
MAX_REDIRECTS = 5


def release_url(cfg, channel, asset):
    if cfg.UPDATE_URL:
        return cfg.UPDATE_URL.format(channel=channel, asset=asset)
    if channel == "stable":
        return "https://github.com/%s/releases/latest/download/%s" % (cfg.UPDATE_REPO, asset)
    return "https://github.com/%s/releases/download/firmware-%s/%s" % (cfg.UPDATE_REPO, channel, asset)


def _split(url):
    scheme, _, rest = url.partition("://")
    hostport, _, path = rest.partition("/")
    host, _, port = hostport.partition(":")
    tls = scheme == "https"
    return tls, host, int(port) if port else (443 if tls else 80), "/" + path


async def http_get(url):
    """GET url following redirects. Returns (stream, content_length)."""
    for _ in range(MAX_REDIRECTS + 1):
        tls, host, port, path = _split(url)
        ip = await dns.resolve(host)
        ctx = None
        if tls:
            import ssl

            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            ctx.verify_mode = ssl.CERT_NONE
        if ctx is None:
            r, w = await asyncio.wait_for(asyncio.open_connection(ip, port), TIMEOUT)
        else:
            r, w = await asyncio.wait_for(asyncio.open_connection(ip, port, ssl=ctx, server_hostname=host), TIMEOUT)
        try:
            req = "GET %s HTTP/1.0\r\nHost: %s\r\nUser-Agent: tree-firmware\r\nAccept: */*\r\nConnection: close\r\n\r\n" % (path, host)
            w.write(req.encode())
            await asyncio.wait_for(w.drain(), TIMEOUT)
            status = await asyncio.wait_for(r.readline(), TIMEOUT)
            parts = status.split(None, 2)
            code = int(parts[1]) if len(parts) > 1 else 0
            headers = {}
            while True:
                line = await asyncio.wait_for(r.readline(), TIMEOUT)
                line = line.strip()
                if not line:
                    break
                k, _, v = line.decode().partition(":")
                headers[k.strip().lower()] = v.strip()
        except BaseException:
            w.close()
            raise
        if code in (301, 302, 303, 307, 308) and "location" in headers:
            w.close()
            loc = headers["location"]
            if loc.startswith("/"):
                loc = "%s://%s%s" % ("https" if tls else "http", host if port in (80, 443) else "%s:%d" % (host, port), loc)
            url = loc
            continue
        if code != 200:
            w.close()
            raise OSError("HTTP %d for %s" % (code, url))
        if "chunked" in headers.get("transfer-encoding", ""):
            w.close()
            raise OSError("chunked responses are not supported")
        length = headers.get("content-length")
        return r, int(length) if length is not None else None
    raise OSError("too many redirects")


class Updates:
    def __init__(self, rt):
        self.rt = rt
        self.state = "idle"
        self.error = None
        self.progress = None
        self.total = None
        self.source = None
        self.installed = None
        self.latest = None
        self.latest_checked = None
        self.check_error = None
        self._check_now = False
        self.busy = False
        self.hb = time.ticks_ms()

    def self_update_enabled(self):
        return bool(self.rt.cfg.UPDATE_KEY)

    def status(self):
        return {
            "state": self.state,
            "source": self.source,
            "progress": self.progress,
            "total": self.total,
            "error": self.error,
            "installed": self.installed,
            "latest": self.latest,
            "channel": self.rt.cfg.UPDATE_CHANNEL,
            "self_update": self.self_update_enabled(),
            "check_error": self.check_error,
        }

    def ha_payload(self):
        cur = self.rt.version
        pct = None
        if self.busy and self.total:
            pct = min(100, (self.progress or 0) * 100 // self.total)
        return {
            "installed_version": cur,
            "latest_version": self.latest or cur,
            "title": "Tree firmware (%s)" % self.rt.cfg.UPDATE_CHANNEL,
            "release_url": "https://github.com/%s/releases" % self.rt.cfg.UPDATE_REPO,
            "in_progress": self.busy,
            "update_percentage": pct,
        }

    def _set(self, state, error=None):
        self.state = state
        self.error = error
        if self.rt.ha:
            self.rt.ha.mark("update")

    async def install_stream(self, read_chunk, length, source, require_signature=False, dry_run=False):
        """Feed a bundle from read_chunk() (async, returns bytes or None at
        the end) into the inactive slot. Returns the manifest. Raises
        bootmgr.InstallError (or the reader's error) on any problem; the
        slot is wiped and nothing is switched."""
        if self.busy:
            raise bootmgr.InstallError("another update is in progress")
        self.busy = True
        self.source = source
        self.total = length
        self.progress = 0
        self._set("receiving")
        inst = None
        try:
            inst = bootmgr.Installer(
                length, self.rt.cfg.UPDATE_KEY, require_signature=require_signature, allow_during_trial=dry_run
            )
            while True:
                self.hb = time.ticks_ms()
                chunk = await read_chunk()
                if chunk is None:
                    break
                inst.feed(chunk)
                self.progress = inst.got
            self._set("verifying")
            for _ in inst.finish():
                self.hb = time.ticks_ms()
                await asyncio.sleep_ms(0)
            if dry_run:
                inst.abort()
                self._set("verified")
                return inst.manifest
            bootmgr.commit(inst.slot)
            self.installed = inst.manifest.get("version")
            self._set("installed")
            log("update: installed %s from %s into slot %s" % (self.installed, source, inst.slot))
            return inst.manifest
        except BaseException as e:
            if inst is not None:
                inst.abort()
            self._set("error", str(e) or repr(e))
            log("update from %s failed: %r" % (source, e))
            raise
        finally:
            self.busy = False

    # -- GitHub --

    async def fetch_manifest(self, channel):
        r, length = await http_get(release_url(self.rt.cfg, channel, MANIFEST))
        try:
            if length is not None and length > bootmgr.MAX_MANIFEST:
                raise OSError("manifest too large")
            data = bytearray()
            while len(data) <= bootmgr.MAX_MANIFEST:
                chunk = await asyncio.wait_for(r.read(1024), TIMEOUT)
                if not chunk:
                    break
                data.extend(chunk)
            return json.loads(data)
        finally:
            r.close()

    async def check(self, channel=None):
        channel = channel or self.rt.cfg.UPDATE_CHANNEL
        try:
            m = await self.fetch_manifest(channel)
            self.latest = m.get("version")
            self.check_error = None
        except Exception as e:
            self.check_error = repr(e)
            log("update: check failed: %r" % (e,))
        self.latest_checked = time.ticks_ms()
        if self.rt.ha:
            self.rt.ha.mark("update")
        return {"channel": channel, "installed": self.rt.version, "latest": self.latest, "error": self.check_error,
                "available": bool(self.latest and self.latest != self.rt.version)}

    def check_soon(self):
        self._check_now = True

    def start_github(self, channel):
        """Start a self-update in the background. Returns an error string."""
        if channel not in CHANNELS:
            return "unknown channel " + str(channel)
        if not self.self_update_enabled():
            return "self-update needs UPDATE_KEY in config.py (bundles must be signed)"
        if self.busy:
            return "an update is already in progress"
        if not self.rt.wifi.up:
            return "wifi is down"
        asyncio.create_task(self._github(channel))
        return None

    async def _github(self, channel):
        self._set("downloading")
        try:
            r, length = await http_get(release_url(self.rt.cfg, channel, ASSET))
        except Exception as e:
            self._set("error", "download failed: %r" % (e,))
            return
        if length is None:
            r.close()
            self._set("error", "server did not send Content-Length")
            return
        state = {"left": length}

        async def read_chunk():
            if state["left"] <= 0:
                return None
            chunk = await asyncio.wait_for(r.read(min(2048, state["left"])), TIMEOUT)
            if not chunk:
                raise bootmgr.InstallError("download truncated")
            state["left"] -= len(chunk)
            return chunk

        try:
            await self.install_stream(read_chunk, length, "github:" + channel, require_signature=True)
        except Exception:
            return
        finally:
            r.close()
        self.rt.request_reboot("installed update from GitHub")

    async def run(self):
        """Periodic update check (for the HA update entity)."""
        hours = self.rt.cfg.UPDATE_CHECK_HOURS
        first = True
        while True:
            self.hb = time.ticks_ms()
            await asyncio.sleep_ms(1000)
            if not self.rt.wifi.up or self.busy or self.rt.calib.playing():
                continue
            due = self._check_now
            if hours and not due:
                if self.latest_checked is None:
                    due = first and time.ticks_diff(time.ticks_ms(), self.rt.started) > 60000
                else:
                    due = time.ticks_diff(time.ticks_ms(), self.latest_checked) > hours * 3600000
            if due:
                self._check_now = False
                first = False
                await self.check()
