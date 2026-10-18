"""Asynchronous DNS A-record lookup over UDP.

socket.getaddrinfo() blocks until lwIP's resolver gives up, which can exceed
the watchdog when the DNS server is unreachable. This sends the query itself
and polls a non-blocking socket.
"""

import socket
import struct
import time

import asyncio

try:
    import network
except ImportError:  # pragma: no cover
    network = None

_cache = {}


def is_ip(host):
    parts = host.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or int(p) > 255:
            return False
    return True


def _server():
    try:
        return network.WLAN(network.STA_IF).ifconfig()[3]
    except Exception:
        return "8.8.8.8"


def _skip_name(data, pos):
    while True:
        ln = data[pos]
        if ln == 0:
            return pos + 1
        if ln & 0xC0 == 0xC0:
            return pos + 2
        pos += ln + 1


def _query(host, qid):
    q = bytearray(struct.pack(">HHHHHH", qid, 0x0100, 1, 0, 0, 0))
    for label in host.split("."):
        q.append(len(label))
        q.extend(label.encode())
    q.extend(b"\x00\x00\x01\x00\x01")
    return q


def parse(data, qid):
    rid, flags, qd, an, _ns, _ar = struct.unpack(">HHHHHH", data[:12])
    if rid != qid or not flags & 0x8000:
        return None
    if flags & 0x000F:
        raise OSError("dns error %d" % (flags & 0x0F))
    pos = 12
    for _ in range(qd):
        pos = _skip_name(data, pos) + 4
    for _ in range(an):
        pos = _skip_name(data, pos)
        rtype, _cls, ttl, rdlen = struct.unpack(">HHIH", data[pos : pos + 10])
        pos += 10
        if rtype == 1 and rdlen == 4:
            return "%d.%d.%d.%d" % tuple(data[pos : pos + 4]), ttl
        pos += rdlen
    raise OSError("no A record")


async def resolve(host, timeout_ms=4000, server=None):
    if is_ip(host):
        return host
    hit = _cache.get(host)
    if hit and time.ticks_diff(hit[1], time.ticks_ms()) > 0:
        return hit[0]
    import random

    qid = random.getrandbits(16)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setblocking(False)
    try:
        addr = socket.getaddrinfo(server or _server(), 53)[0][-1]
        q = _query(host, qid)
        start = time.ticks_ms()
        sent = 0
        while time.ticks_diff(time.ticks_ms(), start) < timeout_ms:
            if time.ticks_diff(time.ticks_ms(), start) >= sent * 1000:
                s.sendto(q, addr)
                sent += 1
            try:
                data = s.recv(512)
            except OSError:
                await asyncio.sleep_ms(20)
                continue
            res = parse(data, qid)
            if res:
                ip, ttl = res
                _cache[host] = (ip, time.ticks_add(time.ticks_ms(), min(ttl, 3600) * 1000))
                return ip
        raise OSError("dns timeout for " + host)
    finally:
        s.close()
