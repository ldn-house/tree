"""Run the real firmware on the MicroPython unix port as a simulated Pico.

A Device is a directory standing in for the Pico's flash. Each "boot" runs
boot.py then main.py in one MicroPython process with tests/firmware/mocks
ahead of the built-in modules. The harness plays the hardware:

- machine.reset() ends the process (exit 42); the next boot sees a watchdog
  reset cause, as on the RP2 where reset() goes through the watchdog;
- the watchdog: once the firmware has fed it, if feeds stop for longer than
  the WDT timeout the process is killed and the board "resets";
- a firmware that exits to the REPL with the watchdog running is reset by
  it; without the watchdog running, the board would sit at the REPL, and the
  harness records that as a hang;
- Wi-Fi: set_net("up"|"down"|"badauth") changes the mocked link live.

Console output of every boot goes to <root>/console.log.
"""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
MOCKS = HERE / "mocks"
RUNNER = HERE / "runner.py"
RECOVERY = ("boot.py", "main.py", "bootmgr.py", "safemode.py", "cfgcheck.py")
MICROPYTHON = os.environ.get("MICROPYTHON", "micropython")
HEAP = os.environ.get("DEVSIM_HEAP", "1M")


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def mp_env(root: Path, extra=None):
    env = dict(os.environ)
    env["DEVSIM_MOCKS"] = str(MOCKS)
    env["DEVSIM_WDT"] = str(root / ".wdt")
    env["DEVSIM_NET"] = str(root / ".net")
    env["DEVSIM_SCRATCH"] = str(root / ".scratch_regs")
    env.update(extra or {})
    return env


def mp_exec(root: Path, code: str, timeout=60):
    """Run a snippet in the device directory with the mocks (a USB REPL)."""
    script = f"import sys\nsys.path.insert(1, {str(MOCKS)!r})\n{code}\n"
    return subprocess.run(
        [MICROPYTHON, "-X", f"heapsize={HEAP}", "-c", script],
        cwd=root,
        env=mp_env(root),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


class Device:
    def __init__(self, root: Path, port: int | None = None, token: str = "test-token", **config):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.port = port or free_port()
        self.token = token
        self.config = {
            "WIFI_SSID": "testnet",
            "WIFI_PASSWORD": "pw",
            "HOSTNAME": "tree",
            "HTTP_PORT": self.port,
            "HTTP_TOKEN": token,
            "LED_COUNT": 500,
            "TRIAL_HEALTHY_S": 5,
            "TRIAL_DEADLINE_S": 25,
            "UPDATE_CHECK_HOURS": 0,
        }
        self.config.update(config)
        self.boots = []  # (cause, outcome)
        self.proc = None
        self._thread = None
        self._stop = threading.Event()
        self.hung = False
        self.set_net("up")

    # -- flash contents --

    def write_config(self, **changes):
        self.config.update(changes)
        lines = [f"{k} = {v!r}" for k, v in self.config.items() if v is not None]
        (self.root / "config.py").write_text("\n".join(lines) + "\n")

    def install_recovery(self):
        for f in RECOVERY:
            shutil.copy(ROOT / "firmware" / f, self.root / f)
        self.write_config()

    def usb_install(self, bundle: Path):
        shutil.copy(bundle, self.root / "_usb.bin")
        r = mp_exec(self.root, "import bootmgr\nbootmgr.install_file('_usb.bin')\nimport os\nos.remove('_usb.bin')")
        if r.returncode != 0:
            raise RuntimeError(f"USB install failed:\n{r.stdout}\n{r.stderr}")

    def boot_state(self):
        try:
            return json.loads((self.root / "boot.json").read_text())
        except (OSError, ValueError):
            return None

    def set_net(self, mode: str):
        (self.root / ".net").write_text(mode)

    def console(self):
        try:
            return (self.root / "console.log").read_text(errors="replace")
        except OSError:
            return ""

    # -- power --

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self.proc and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait()
        if self._thread:
            self._thread.join(timeout=10)

    def kill_now(self, cause="pwron"):
        """Pull the plug: the next boot is a power-on reset."""
        self._next_cause = cause
        if self.proc and self.proc.poll() is None:
            self.proc.kill()

    def _wdt(self):
        """(last feed time, timeout) or (None, None) before the first feed.
        Uses the file's mtime: reading its contents could catch the mock's
        write mid-truncate and look like a feed at time 0."""
        try:
            fed = (self.root / ".wdt").stat().st_mtime
            timeout = int((self.root / ".wdt.timeout").read_text()) / 1000
            return fed, timeout
        except (OSError, ValueError):
            return None, None

    def interrupt(self):
        """Ctrl-C at the USB REPL, as mpremote does; the runner then calls
        machine.reset(), as mpremote's reset command does."""
        import signal

        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGINT)

    def _loop(self):
        cause = "pwron"
        reason = 0
        (self.root / ".scratch_regs").unlink(missing_ok=True)  # power-on clears them
        log = open(self.root / "console.log", "a")
        while not self._stop.is_set():
            for f in (".wdt", ".wdt.timeout"):
                (self.root / f).unlink(missing_ok=True)
            self._next_cause = None
            log.write(f"\n===== boot {len(self.boots) + 1} (reset cause: {cause}) =====\n")
            log.flush()
            self.proc = subprocess.Popen(
                [MICROPYTHON, "-X", f"heapsize={HEAP}", str(RUNNER)],
                cwd=self.root,
                env=mp_env(self.root, {"DEVSIM_RESET": cause, "DEVSIM_WDT_REASON": str(reason)}),
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            outcome = None
            while outcome is None and not self._stop.is_set():
                time.sleep(0.1)
                rc = self.proc.poll()
                fed, timeout = self._wdt()
                if self._next_cause and rc is not None:
                    outcome, cause, reason = "power cut", self._next_cause, 0
                    (self.root / ".scratch_regs").unlink(missing_ok=True)
                elif rc is None:
                    if fed is not None and time.time() - fed > timeout:
                        self.proc.kill()
                        self.proc.wait()
                        outcome, cause, reason = "watchdog reset", "wdt", 1
                elif rc == 42:
                    outcome, cause, reason = "machine.reset()", "wdt", 2
                elif self._stop.is_set():
                    break  # we killed it (stop()): not a boot outcome
                elif fed is not None:
                    # Dropped to the REPL; the running watchdog resets the board.
                    while time.time() - fed <= timeout and not self._stop.is_set():
                        time.sleep(0.1)
                    outcome, cause, reason = f"exited ({rc}) then watchdog reset", "wdt", 1
                else:
                    outcome = f"exited ({rc}) with no watchdog: board would hang at the REPL"
                    self.hung = True
                    self._stop.set()
            if outcome and not self._stop.is_set():
                self.boots.append(outcome)
                log.write(f"===== {outcome} =====\n")
                log.flush()
        log.close()

    # -- HTTP --

    def url(self, path):
        return f"http://127.0.0.1:{self.port}{path}"

    def request(self, method, path, body=None, token=True, headers=None, timeout=10):
        h = dict(headers or {})
        if token:
            h["Authorization"] = f"Bearer {self.token if token is True else token}"
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            h.setdefault("Content-Type", "application/json")
        req = urllib.request.Request(self.url(path), data=body, method=method, headers=h)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read()
                code = r.status
        except urllib.error.HTTPError as e:
            raw = e.read()
            code = e.code
        try:
            return code, json.loads(raw) if raw else None
        except ValueError:
            return code, raw

    def get(self, path, **kw):
        return self.request("GET", path, token=kw.pop("token", False), **kw)

    def post(self, path, body=b"", **kw):
        return self.request("POST", path, body=body, **kw)

    def wait_http(self, timeout=30, pred=None):
        end = time.time() + timeout
        last = None
        while time.time() < end:
            try:
                code, st = self.get("/", timeout=2)
                if code == 200 and (pred is None or pred(st)):
                    return st
                last = st
            except (OSError, urllib.error.URLError):
                pass
            time.sleep(0.3)
        raise TimeoutError(f"device not ready (last status: {last})\n--- console ---\n{self.console()[-4000:]}")


def wait_for(pred, timeout=30, interval=0.2, what="condition"):
    end = time.time() + timeout
    while time.time() < end:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    raise TimeoutError(f"timed out waiting for {what}")


def build_bundle(out: Path, effects: Path | None = None, key: str | None = None, version: str | None = None,
                 source=False, app: Path | None = None, generation: int | None = None):
    import sys

    cmd = [sys.executable, str(ROOT / "tools" / "build_firmware.py"), "--out", str(out)]
    cmd += ["--effects", str(effects or HERE / "fixtures" / "effects")]
    if key:
        cmd += ["--key", key]
    if version:
        cmd += ["--version", version]
    if source:
        cmd.append("--source")
    app = app or os.environ.get("DEVSIM_APP")  # e.g. an older firmware/app to show a regression
    if app:
        cmd += ["--app", str(app)]
    if generation is not None:
        cmd += ["--generation", str(generation)]
    subprocess.run(cmd, check=True, capture_output=True)
    return out / "tree-firmware.bin"
