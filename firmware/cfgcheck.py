"""Config schema: one validator for every setting, shared by the recovery
layer (bootmgr, safemode), the app (cfg, driver) and the host (flash.py).
Part of the recovery layer (installed over USB). Runs on MicroPython and
CPython.

validate(values) returns (clean, errors). A config with any error is not
used: the device falls back to the last config that was confirmed to reach
the network (bootmgr). Every check is about values the firmware would act
on: ports, timings, the power model and credentials.
"""

DEFAULT_TIMING_NS = (400, 850, 800, 450)
CYCLE_NS = 50  # the PIO program runs at 20 MHz
MAX_CYCLES = 16  # 1 + 4-bit delay

REQUIRED = ("WIFI_SSID", "HTTP_TOKEN")


class ConfigError(ValueError):
    pass


def _finite(v):
    return v == v and v not in (float("inf"), float("-inf"))


def _num(v, lo, hi, integer=False):
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ConfigError("must be a number")
    if integer and not isinstance(v, int):
        raise ConfigError("must be a whole number")
    if not _finite(v) or not lo <= v <= hi:
        raise ConfigError("must be between %s and %s" % (lo, hi))
    return v


def _int(lo, hi):
    return lambda v: _num(v, lo, hi, integer=True)


def _dither(v):
    # "spatial" (fixed per LED, the default), "temporal" (flickers at low
    # levels at ~40 fps), or False. True is accepted and means "spatial".
    if v is True:
        return "spatial"
    if v is False or v in ("spatial", "temporal"):
        return v
    raise ConfigError("must be 'spatial', 'temporal' or False")


def _white(v):
    if v is None:
        return None
    if not isinstance(v, (tuple, list)) or len(v) != 3:
        raise ConfigError("must be three numbers (r, g, b), each 0 to 1")
    return tuple(_num(c, 0.0, 1.0) for c in v)


def _number(lo, hi):
    return lambda v: _num(v, lo, hi)


def _str(lo, hi, none=False, forbid=""):
    def check(v):
        if v is None and none:
            return None
        if not isinstance(v, str) or not lo <= len(v) <= hi:
            raise ConfigError("must be a string of %d to %d characters" % (lo, hi))
        for c in forbid:
            if c in v:
                raise ConfigError("must not contain %r" % c)
        return v

    return check


def _optional(check):
    return lambda v: None if v is None else check(v)


def _choice(*options):
    def check(v):
        if v not in options:
            raise ConfigError("must be one of %s" % ", ".join(repr(o) for o in options))
        return v

    return check


def _hostname(v):
    v = _str(1, 32)(v)
    for c in v:
        if not ("a" <= c.lower() <= "z" or "0" <= c <= "9" or c == "-"):
            raise ConfigError("may only contain letters, digits and -")
    return v


def _color_order(v):
    if not isinstance(v, str) or len(v) != 3 or sorted(v) != ["B", "G", "R"]:
        raise ConfigError("must be a permutation of RGB, e.g. 'RGB' or 'GRB'")
    return v


def _url(v):
    v = _str(10, 256)(v)
    if not (v.startswith("http://") or v.startswith("https://")) or "{asset}" not in v:
        raise ConfigError("must be an http(s) URL template containing {asset}")
    return v


def _gamma(v):
    if v is None:
        return None
    # Gamma >= 1 never raises a channel above its linear value, so the
    # power cap (applied before gamma) still holds on the wire.
    return _num(v, 1.0, 4.0)


def timing_cycles(timing_ns):
    """PIO cycle counts (T1, T2, T3) for (T0H, T0L, T1H, T1L) in ns.

    The program holds the line high for T1 then, for a 1 bit, T2 more; low
    for T3, plus T2 for a 0 bit: T0H = T1, T1H = T1 + T2, T1L = T3,
    T0L = T2 + T3, so T0H + T0L must equal T1H + T1L. Rounded to 50 ns."""
    try:
        if len(timing_ns) != 4:
            raise TypeError()
        t0h, t0l, t1h, t1l = (int(v) for v in timing_ns)
        for v in timing_ns:
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not _finite(v):
                raise TypeError()
    except (TypeError, ValueError):
        raise ConfigError("LED_TIMING_NS must be four numbers (T0H, T0L, T1H, T1L) in ns")
    if t0h + t0l != t1h + t1l:
        raise ConfigError("LED_TIMING_NS: T0H + T0L must equal T1H + T1L (one bit period)")
    c1 = (t0h + CYCLE_NS // 2) // CYCLE_NS
    c2 = (t1h - t0h + CYCLE_NS // 2) // CYCLE_NS
    c3 = (t1l + CYCLE_NS // 2) // CYCLE_NS
    for c in (c1, c2, c3):
        if not 1 <= c <= MAX_CYCLES:
            raise ConfigError("LED_TIMING_NS out of range (50-800 ns per step, T1H > T0H)")
    return c1, c2, c3


def _timing(v):
    if v is None:
        return None
    timing_cycles(v)
    return tuple(int(x) for x in v)


def timing_or_default(v):
    """A timing safe to hand to the PIO program or machine.bitstream."""
    try:
        t = _timing(v)
        return t if t is not None else DEFAULT_TIMING_NS
    except ConfigError:
        return DEFAULT_TIMING_NS


SCHEMA = {
    "WIFI_SSID": _str(1, 32),
    "WIFI_PASSWORD": _str(0, 64, none=True),
    "HOSTNAME": _hostname,
    "HTTP_PORT": _int(1, 65535),
    "OTA_PORT": _int(1, 65535),
    "HTTP_TOKEN": _str(8, 128, forbid=" \t\r\n"),
    "MQTT_BROKER": _str(1, 253, none=True, forbid=" /"),
    "MQTT_PORT": _int(1, 65535),
    "MQTT_USER": _str(0, 128, none=True),
    "MQTT_PASSWORD": _str(0, 128, none=True),
    "MQTT_PREFIX": _str(1, 64, forbid="#+ "),
    "MQTT_KEEPALIVE": _int(5, 600),
    "HA_DISCOVERY_PREFIX": _str(1, 64, forbid="#+ "),
    "UPDATE_KEY": _str(16, 256, none=True),
    "UPDATE_CHANNEL": _choice("main", "stable"),
    "UPDATE_REPO": _str(3, 100, forbid=" "),
    "UPDATE_URL": _optional(_url),
    "UPDATE_CHECK_HOURS": _int(0, 168),
    "UPDATE_CHECK_FIRST_S": _int(0, 86400),
    "LED_COUNT": _int(1, 2000),
    "LED_PIN": _int(0, 29),
    "COLOR_ORDER": _color_order,
    "LED_TIMING_NS": _timing,
    # Amps at 12 V; 0 is allowed and means dark.
    "POWER_LIMIT_A": _number(0, 20),
    # mA per channel at full duty: strictly positive, or the cap means nothing.
    "LED_MA_PER_CHANNEL": _number(0.5, 100),
    "POWER_BUDGET_PERCENT": _optional(_number(0, 100)),
    "GAMMA": _gamma,
    "LED_DITHER": _dither,
    "WHITE_BALANCE": _white,
    "FPS": _int(1, 100),
    "TRIAL_HEALTHY_S": _int(1, 3600),
    "TRIAL_DEADLINE_S": _int(5, 7200),
}


def validate(values, strict=False):
    """(clean, errors). strict: unknown settings are errors (uploads, where
    a typo would otherwise be silently ignored); otherwise they are dropped."""
    clean = {}
    errors = []
    if not isinstance(values, dict):
        return clean, ["config must be an object"]
    for k, v in values.items():
        check = SCHEMA.get(k)
        if check is None:
            if strict:
                errors.append("%s: unknown setting" % (k,))
            continue
        try:
            clean[k] = check(v)
        except ConfigError as e:
            errors.append("%s %s" % (k, e))
    for k in REQUIRED:
        if k not in clean:
            errors.append("%s is required" % k)
    return clean, errors
