"""network mock. The link state comes from the file named by $DEVSIM_NET,
read on every call, so a test can take Wi-Fi up and down at runtime:
    up        associates ~0.5 s after connect()
    down      connect() never completes (AP missing)
    badauth   connect() fails with STAT_WRONG_PASSWORD
"""

import os
import time

STA_IF = 0
AP_IF = 1
STAT_IDLE = 0
STAT_CONNECTING = 1
STAT_WRONG_PASSWORD = -3
STAT_NO_AP_FOUND = -2
STAT_CONNECT_FAIL = -1
STAT_GOT_IP = 3

_hostname = "mock"


def hostname(name=None):
    global _hostname
    if name is None:
        return _hostname
    _hostname = name


def _mode():
    path = os.getenv("DEVSIM_NET")
    if not path:
        return "up"
    try:
        with open(path) as f:
            return f.read().strip() or "up"
    except OSError:
        return "up"


class WLAN:
    _inst = None

    def __new__(cls, iface=STA_IF):
        if WLAN._inst is None:
            WLAN._inst = object.__new__(cls)
            WLAN._inst._active = False
            WLAN._inst._connect_at = None
        return WLAN._inst

    def __init__(self, iface=STA_IF):
        pass

    def active(self, v=None):
        if v is None:
            return self._active
        self._active = bool(v)
        if not v:
            self._connect_at = None

    def config(self, *a, **kw):
        return None

    def connect(self, ssid=None, password=None):
        if not self._active:
            raise OSError("wifi not active")
        self._connect_at = time.ticks_ms()

    def disconnect(self):
        self._connect_at = None

    def isconnected(self):
        return self.status() == STAT_GOT_IP

    def status(self, param=None):
        if param == "rssi":
            return -55
        if not self._active or self._connect_at is None:
            return STAT_IDLE
        mode = _mode()
        if mode == "badauth":
            return STAT_WRONG_PASSWORD
        if mode != "up":
            return STAT_CONNECTING
        if time.ticks_diff(time.ticks_ms(), self._connect_at) < 500:
            return STAT_CONNECTING
        return STAT_GOT_IP

    def ifconfig(self):
        return ("127.0.0.1", "255.255.255.0", "127.0.0.1", "127.0.0.1")
