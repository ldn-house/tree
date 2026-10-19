"""Safe mode: the recovery server. Part of the recovery layer (USB only).

Entered by bootmgr when no slot can boot. It blanks the LEDs, joins Wi-Fi and
serves a deliberately small, synchronous HTTP API on the usual port:

    GET  /                 status: why we are here, boot state, last crash
    POST /update           install a firmware bundle (same format and auth
                           as the normal firmware), then reboot into it
    POST /recovery/retry   leave safe mode and try the installed firmware
    POST /reboot           reboot (stays in safe mode)

Mutating requests need the HTTP_TOKEN from config.py. Every loop feeds the
watchdog and every socket operation has a timeout, so nothing here can hang
the board. After SAFE_RETRY_S without an update the firmware is retried.
"""

import gc
import json
import socket
import time

import bootmgr

try:
    import machine
    import network
except ImportError:  # pragma: no cover
    machine = None
    network = None

SAFE_RETRY_S = 3600
WIFI_RETRY_S = 30
MAX_HEADER = 4096


def _cfg(name, default=None):
    return bootmgr.config_value(name, default)


def _blank():
    n = _cfg("LED_COUNT", 500)
    try:
        pin = machine.Pin(_cfg("LED_PIN", 0), machine.Pin.OUT)
        machine.bitstream(pin, 0, (400, 850, 800, 450), bytes(3 * n))
    except Exception as e:
        bootmgr.log("safe mode: cannot blank LEDs: %r" % (e,))


class _Wifi:
    def __init__(self):
        self.wlan = None
        self.last = None
        ssid = _cfg("WIFI_SSID")
        if network is None or not ssid:
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
            try:
                self.wlan.connect(_cfg("WIFI_SSID"), _cfg("WIFI_PASSWORD"))
            except Exception as e:
                bootmgr.log("safe mode: wifi connect failed: %r" % (e,))

    def ip(self):
        try:
            return self.wlan.ifconfig()[0] if self.wlan and self.wlan.isconnected() else None
        except Exception:
            return None


def _read_head(c):
    data = b""
    while b"\r\n\r\n" not in data:
        chunk = c.recv(512)
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
    return (parts[0], parts[1].split("?", 1)[0], headers), rest


def _send(c, code, obj):
    body = json.dumps(obj).encode()
    reason = {200: "OK", 400: "Bad Request", 401: "Unauthorized", 403: "Forbidden", 404: "Not Found"}.get(code, "Error")
    c.sendall(
        b"HTTP/1.0 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n"
        % (code, reason.encode(), len(body))
    )
    c.sendall(body)


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
    return {
        "mode": "safe",
        "reason": reason,
        "hostname": _cfg("HOSTNAME", "tree"),
        "ip": wifi.ip(),
        "boot": bootmgr.read_state(),
        "reset_cause": info.get("reset_cause"),
        "crash": crash,
        "free_memory": gc.mem_free(),
    }


def _install(c, headers, rest):
    try:
        length = int(headers.get("content-length", ""))
    except ValueError:
        return 400, {"error": "Content-Length required"}
    try:
        inst = bootmgr.Installer(length, _cfg("UPDATE_KEY"), allow_during_trial=True)
    except bootmgr.InstallError as e:
        return 400, {"error": str(e)}
    try:
        if rest:
            inst.feed(rest)
        while inst.got < length:
            chunk = c.recv(min(2048, length - inst.got))
            if not chunk:
                break
            inst.feed(chunk)
            bootmgr.feed()
        for _ in inst.finish():
            bootmgr.feed()
        bootmgr.commit(inst.slot)
    except Exception as e:  # InstallError, OSError (timeout), MemoryError
        inst.abort()
        return 400, {"error": str(e) or repr(e)}
    return 200, {"installed": inst.manifest.get("version"), "slot": inst.slot, "rebooting": True}


def run(reason, info):
    wdt = bootmgr.watchdog()
    bootmgr.log("safe mode: " + str(reason))
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
            bootmgr.leave_safe_mode()
            bootmgr.reset()
        try:
            c, addr = srv.accept()
        except OSError:
            continue
        reboot = False
        try:
            c.settimeout(5)
            req, rest = _read_head(c)
            if req is None:
                continue
            method, path, headers = req
            if method == "GET" and path in ("/", "/status"):
                _send(c, 200, _status(reason, info, wifi))
            elif method != "POST" or path not in ("/update", "/reboot", "/recovery/retry"):
                _send(c, 404, {"error": "safe mode: not available"})
            elif not _authorized(headers):
                _send(c, 401, {"error": "missing or bad token"})
            elif path == "/update":
                code, obj = _install(c, headers, rest)
                _send(c, code, obj)
                reboot = code == 200
            elif path == "/recovery/retry":
                bootmgr.leave_safe_mode()
                _send(c, 200, {"rebooting": True})
                reboot = True
            else:
                _send(c, 200, {"rebooting": True})
                reboot = True
        except Exception as e:
            bootmgr.log("safe mode request failed: %r" % (e,))
        finally:
            c.close()
            gc.collect()
        if reboot:
            time.sleep_ms(200)
            bootmgr.reset()
