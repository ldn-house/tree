"""Safe mode: the recovery server. Part of the recovery layer.

Entered by bootmgr when no slot can boot, or when the boot can't even be
recorded (flash full or failing). It frees disposable space, blanks the
LEDs, joins Wi-Fi (with the last good config if config.py is broken) and
serves a deliberately small, synchronous HTTP API on the usual port:

    GET  /                 status: why we are here, boot state, last crash
    POST /update           install a firmware bundle (same format and auth
                           as the normal firmware; ?allow_downgrade=1 to
                           accept an older generation), then reboot into it
    PUT  /config           replace config.py from JSON values (signed with
                           UPDATE_KEY when one is set), then reboot
    POST /recovery/retry   leave safe mode and try the installed firmware
    POST /reboot           reboot (stays in safe mode)

Mutating requests need the HTTP_TOKEN. Every loop feeds the watchdog, every
socket read has a short timeout and every request a whole-request deadline,
so a slow or stuck client can neither hang the board nor hold it for long.
After SAFE_RETRY_S without an update the firmware is retried.
"""

import gc
import json
import socket
import time

import bootmgr
import cfgcheck

try:
    import machine
    import network
except ImportError:  # pragma: no cover
    machine = None
    network = None

SAFE_RETRY_S = 3600
WIFI_RETRY_S = 30
WIFI_RESET_EVERY = 4
MAX_HEADER = 4096
MAX_CONFIG = 4096
RECV_TIMEOUT_S = 2
HEAD_DEADLINE_S = 10
UPLOAD_DEADLINE_S = 180


def _cfg(name, default=None):
    return bootmgr.config_value(name, default)


def _blank():
    # Only validated values reach the synchronous C bitstream routine: a bad
    # timing (all zeros never finishes on the RP2350) or LED count must not
    # stop recovery before Wi-Fi starts.
    n = _cfg("LED_COUNT", 500)
    if not (isinstance(n, int) and 1 <= n <= 2000):
        n = 500
    pin_no = _cfg("LED_PIN", 0)
    if not (isinstance(pin_no, int) and 0 <= pin_no <= 29):
        pin_no = 0
    timing = cfgcheck.timing_or_default(_cfg("LED_TIMING_NS"))
    try:
        pin = machine.Pin(pin_no, machine.Pin.OUT)
        machine.bitstream(pin, 0, timing, bytes(3 * n))
    except Exception as e:
        bootmgr.log("safe mode: cannot blank LEDs: %r" % (e,))


class _Wifi:
    def __init__(self):
        self.wlan = None
        self.last = None
        self.attempts = 0
        # Credentials to try in turn: the current config's, then the last
        # config confirmed to reach the network, if different.
        self.creds = []
        for c in (bootmgr.load_config()[0], bootmgr.good_config() or {}):
            pair = (c.get("WIFI_SSID"), c.get("WIFI_PASSWORD"))
            if pair[0] and pair not in self.creds:
                self.creds.append(pair)
        if network is None or not self.creds:
            bootmgr.log("safe mode: no Wi-Fi credentials; USB needed")
            return
        try:
            network.hostname(_cfg("HOSTNAME", "tree"))
        except Exception:
            pass
        self.wlan = network.WLAN(network.STA_IF)
        self.wlan.active(True)

    def poll(self):
        if self.wlan is None or self.wlan.isconnected():
            return
        now = time.time()
        if self.last is None or now - self.last >= WIFI_RETRY_S:
            self.last = now
            self.attempts += 1
            try:
                if self.attempts % WIFI_RESET_EVERY == 0:
                    # Same recovery as the app: power-cycle a stuck interface.
                    self.wlan.active(False)
                    bootmgr.feed()
                    self.wlan.active(True)
                ssid, password = self.creds[(self.attempts // 2) % len(self.creds)]
                self.wlan.disconnect()
                self.wlan.connect(ssid, password)
            except Exception as e:
                bootmgr.log("safe mode: wifi connect failed: %r" % (e,))

    def ip(self):
        try:
            return self.wlan.ifconfig()[0] if self.wlan and self.wlan.isconnected() else None
        except Exception:
            return None


class _Deadline(Exception):
    pass


def _recv(c, n, deadline):
    """recv with the watchdog fed and an overall deadline."""
    bootmgr.feed()
    if time.time() > deadline:
        raise _Deadline()
    return c.recv(n)


def _read_head(c, deadline):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = _recv(c, 512, deadline)
        if not chunk:
            return None, None
        data += chunk
        if len(data) > MAX_HEADER:
            return None, None
    head, rest = data.split(b"\r\n\r\n", 1)
    lines = head.decode().split("\r\n")
    parts = lines[0].split(" ")
    if len(parts) < 2:
        return None, None
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    path, _, query = parts[1].partition("?")
    return (parts[0], path, query, headers), rest


def _send_all(c, data):
    """socket.sendall() isn't in every MicroPython port (1.27's unix port
    lacks it), so loop on send()."""
    mv = memoryview(data)
    while mv:
        bootmgr.feed()
        n = c.send(mv)
        if not n:
            raise OSError("send failed")
        mv = mv[n:]


def _send(c, code, obj):
    body = json.dumps(obj).encode()
    reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found"}.get(code, "Error")
    _send_all(
        c,
        b"HTTP/1.0 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
        % (code, reason.encode(), len(body)),
    )
    _send_all(c, body)


def _authorized(headers):
    token = _cfg("HTTP_TOKEN")
    if not token:
        return False
    given = headers.get("x-tree-token")
    auth = headers.get("authorization", "")
    if auth[:7].lower() == "bearer ":
        given = auth[7:].strip()
    return bool(given) and bootmgr.eq(given.encode(), token.encode())


def _status(reason, info, wifi):
    crash = None
    try:
        with open(bootmgr.CRASH) as f:
            crash = f.read()[-1500:]
    except OSError:
        pass
    src, cerr = bootmgr.config_source()
    return {
        "mode": "safe",
        "reason": reason,
        "hostname": _cfg("HOSTNAME", "tree"),
        "ip": wifi.ip(),
        "boot": bootmgr.read_state(),
        "reset_cause": info.get("reset_cause"),
        "crash": crash,
        "config": {"source": src, "error": cerr},
        "free_memory": gc.mem_free(),
        "free_flash": bootmgr.free_bytes(),
    }


def _length(headers):
    try:
        return int(headers.get("content-length", ""))
    except ValueError:
        return None


def parse_query(qs):
    q = {}
    for part in qs.split("&"):
        if part:
            k, _, v = part.partition("=")
            q[k] = v
    return q


def flag(q, name):
    """Exact key=value, same policy as the app's API (no substring games)."""
    return q.get(name) in ("1", "true", "yes")


def _install(c, headers, query, rest, deadline):
    length = _length(headers)
    if length is None:
        return 400, {"error": "Content-Length required"}
    q = parse_query(query)
    try:
        # reclaim_slot=1: if the flash is too full, explicitly allow the
        # inactive slot to be deleted to make room (only if that suffices).
        inst = bootmgr.Installer(
            length, _cfg("UPDATE_KEY"), allow_downgrade=flag(q, "allow_downgrade"), reclaim_slot=flag(q, "reclaim_slot")
        )
    except bootmgr.InstallError as e:
        return 400, {"error": str(e)}
    try:
        if rest:
            inst.feed(rest)
        while inst.got < length:
            chunk = _recv(c, min(2048, length - inst.got), deadline)
            if not chunk:
                break
            inst.feed(chunk)
        for _ in inst.finish():
            bootmgr.feed()
        slot = bootmgr.commit(allow_during_trial=True)
    except _Deadline:
        inst.abort()
        return 400, {"error": "upload took longer than %d s" % UPLOAD_DEADLINE_S}
    except Exception as e:  # InstallError, OSError (timeout), MemoryError
        inst.abort()
        _drain(c, length - inst.got, deadline)
        return 400, {"error": str(e) or repr(e)}
    return 200, {"installed": inst.manifest.get("version"), "slot": slot, "rebooting": True}


def _drain(c, left, deadline):
    """Read and drop the rest of a rejected upload (within its deadline) so
    the client gets the error rather than a connection reset."""
    try:
        while left > 0:
            chunk = _recv(c, min(2048, left), deadline)
            if not chunk:
                return
            left -= len(chunk)
    except Exception:
        pass


def _config(c, headers, rest, deadline):
    length = _length(headers)
    if length is None or length > MAX_CONFIG:
        return 400, {"error": "Content-Length required (max %d)" % MAX_CONFIG}
    body = rest
    while len(body) < length:
        chunk = _recv(c, length - len(body), deadline)
        if not chunk:
            return 400, {"error": "body truncated"}
        body += chunk
    if not bootmgr.signature_ok("config", body, headers.get("x-signature")):
        return 403, {"error": "config uploads must be signed with UPDATE_KEY (X-Signature)"}
    try:
        keys = bootmgr.install_config(json.loads(body), trial=False)
    except (ValueError, SyntaxError) as e:
        return 400, {"error": str(e)}
    return 200, {"written": keys, "rebooting": True}


def run(reason, info):
    wdt = bootmgr.watchdog()
    bootmgr.log("safe mode: " + str(reason))
    # Make room first: if the flash filled up, this is what lets an
    # upload (and the boot state write that commits it) succeed.
    try:
        bootmgr.reclaim()
    except Exception as e:
        bootmgr.log("safe mode: reclaim failed: %r" % (e,))
    _blank()
    wifi = _Wifi()
    port = _cfg("HTTP_PORT", _cfg("OTA_PORT", 8080))
    srv = socket.socket()
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(socket.getaddrinfo("0.0.0.0", port)[0][-1])
    srv.listen(2)
    srv.settimeout(0.5)
    started = time.time()
    while True:
        if wdt:
            wdt.feed()
        wifi.poll()
        if time.time() - started > SAFE_RETRY_S:
            try:
                bootmgr.leave_safe_mode()
            except Exception as e:
                bootmgr.log("safe mode: %r" % (e,))
            bootmgr.reset()
        try:
            c, addr = srv.accept()
        except OSError:
            continue
        reboot = False
        try:
            c.settimeout(RECV_TIMEOUT_S)
            req, rest = _read_head(c, time.time() + HEAD_DEADLINE_S)
            if req is None:
                continue
            method, path, query, headers = req
            if method == "GET" and path in ("/", "/status"):
                _send(c, 200, _status(reason, info, wifi))
            elif (method, path) not in (("POST", "/update"), ("POST", "/reboot"), ("POST", "/recovery/retry"),
                                        ("PUT", "/config"), ("POST", "/config")):
                _send(c, 404, {"error": "safe mode: not available"})
            elif not _authorized(headers):
                _send(c, 401, {"error": "missing or bad token"})
            elif path == "/update":
                code, obj = _install(c, headers, query, rest, time.time() + UPLOAD_DEADLINE_S)
                _send(c, code, obj)
                reboot = code == 200
            elif path == "/config":
                code, obj = _config(c, headers, rest, time.time() + HEAD_DEADLINE_S)
                _send(c, code, obj)
                reboot = code == 200
            elif path == "/recovery/retry":
                bootmgr.leave_safe_mode()
                _send(c, 200, {"rebooting": True})
                reboot = True
            else:
                _send(c, 200, {"rebooting": True})
                reboot = True
        except Exception as e:  # timeouts, _Deadline, resets
            bootmgr.log("safe mode request failed: %r" % (e,))
        finally:
            c.close()
            gc.collect()
        if reboot:
            time.sleep_ms(200)
            bootmgr.reset()
