"""Effect discovery and loading (docs/effect-api.md).

Effects come from the slot's effects/ package, listed in effects.EFFECTS.
An effect can also be hot-pushed over HTTP without a reboot (PUT
/effects/<name>); it is stored in /fx/fx_<name>.py (or .mpy) and takes
precedence over the packaged module of the same name. Overlays are cleared
when new firmware is installed.
"""

import json
import os
import sys

import bootmgr

from app.log import log

OVERLAY = bootmgr.FX_DIR
PREFIX = "fx_"


def valid_name(name):
    if not isinstance(name, str) or not 1 <= len(name) <= 32 or name == "auto":
        return False
    if not ("a" <= name[0] <= "z") or "__" in name:
        return False
    for ch in name:
        if not ("a" <= ch <= "z" or "0" <= ch <= "9" or ch == "_"):
            return False
    return True


def module_meta(mod):
    params = {}
    for k, spec in getattr(mod, "PARAMS", {}).items():
        try:
            params[k] = (spec[0], spec[1], spec[2])
        except Exception:
            pass
    return {
        "duration_ms": int(getattr(mod, "DURATION_MS", 20000)),
        "params": params,
        "needs_coords": bool(getattr(mod, "NEEDS_COORDS", False)),
    }


STAGING = "__new"


def clean_staging():
    """Remove half-validated pushes (the board reset mid-validation)."""
    try:
        for f in os.listdir(OVERLAY):
            if STAGING in f:
                os.remove(OVERLAY + "/" + f)
                log("fx: removed unvalidated push " + f)
    except OSError:
        pass


class Effects:
    def __init__(self, slot_dir=None):
        if OVERLAY not in sys.path:
            sys.path.append(OVERLAY)
        clean_staging()
        self.base = []
        self.error = None
        try:
            import effects

            self.base = [n for n in effects.EFFECTS if valid_name(n)]
        except Exception as e:
            self.error = repr(e)
            log("fx: no effects package: %r" % (e,))
        self._meta = {}
        if slot_dir:
            try:
                with open(slot_dir + "/data/effects.json") as f:
                    data = json.load(f)
                for name, m in data.items():
                    params = {}
                    for k, spec in m.get("params", {}).items():
                        params[k] = tuple(spec)
                    m["params"] = params
                    self._meta[name] = m
            except Exception:
                pass
        self._names = None

    def overlays(self):
        out = []
        try:
            for f in os.listdir(OVERLAY):
                if f.startswith(PREFIX) and (f.endswith(".py") or f.endswith(".mpy")):
                    name = f[len(PREFIX) : f.rindex(".")]
                    if valid_name(name) and STAGING not in name and name not in out:
                        out.append(name)
        except OSError:
            pass
        out.sort()
        return out

    def names(self):
        if self._names is None:
            names = list(self.base)
            for n in self.overlays():
                if n not in names:
                    names.append(n)
            self._names = names
        return self._names

    def changed(self):
        self._names = None

    def overlay_file(self, name):
        for ext in (".mpy", ".py"):
            p = OVERLAY + "/" + PREFIX + name + ext
            if bootmgr.exists(p):
                return p
        return None

    def _modname(self, name):
        if self.overlay_file(name):
            return PREFIX + name
        return "effects." + name

    def meta(self, name):
        if self.overlay_file(name) is None and name in self._meta:
            return self._meta[name]
        try:
            mod = self.load(name)
        except Exception as e:
            log("fx: cannot read metadata for %s: %r" % (name, e))
            return {"duration_ms": 20000, "params": {}, "needs_coords": False}
        m = module_meta(mod)
        self._meta[name] = m
        return m

    def load(self, name):
        if name not in self.names():
            raise ValueError("unknown effect " + name)
        modname = self._modname(name)
        if modname.startswith(PREFIX):
            return __import__(modname)
        pkg = __import__(modname)
        return getattr(pkg, name)

    def unload(self, name):
        for modname in (PREFIX + name, "effects." + name):
            if modname in sys.modules:
                del sys.modules[modname]
        try:
            import effects

            if hasattr(effects, name):
                delattr(effects, name)
        except Exception:
            pass

    def forget(self, name):
        self.unload(name)
        if name in self._meta:
            del self._meta[name]
        self.changed()
