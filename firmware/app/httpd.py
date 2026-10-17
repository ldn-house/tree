"""Minimal asyncio HTTP/1.0 server with token auth.

Every request runs in its own task with timeouts on the request head and on
each body chunk, so a slow or vanished client never stalls anything else
(review L1). Responses are JSON built with json.dumps (proper escaping) and
written with drain(). Mutating routes need the shared token from config.py,
sent as "Authorization: Bearer <token>" or "X-Tree-Token: <token>"; without
a configured token they are refused (fail closed, review H11).
"""

import json
import time

import asyncio

import bootmgr

from app.log import log

HEAD_TIMEOUT = 5
CHUNK_TIMEOUT = 10
MAX_LINE = 1024
MAX_HEADERS = 32
MAX_CONNECTIONS = 4

REASONS = {
    200: "OK",
    202: "Accepted",
    204: "No Content",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    409: "Conflict",
    411: "Length Required",
    413: "Payload Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}

CORS = (
    "Access-Control-Allow-Origin: *\r\n"
    "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
    "Access-Control-Allow-Headers: Content-Type, Authorization, X-Tree-Token\r\n"
    "Access-Control-Allow-Private-Network: true\r\n"
)


class HTTPError(Exception):
    def __init__(self, code, msg):
        super().__init__(msg)
        self.code = code
        self.msg = msg


def unquote(s):
    s = s.replace("+", " ")
    if "%" not in s:
        return s
    out = bytearray()
    i = 0
    b = s.encode()
    while i < len(b):
        if b[i] == 0x25 and i + 2 < len(b):
            try:
                out.append(int(b[i + 1 : i + 3], 16))
                i += 3
                continue
            except ValueError:
                pass
        out.append(b[i])
        i += 1
    return out.decode()


def parse_query(qs):
    q = {}
    for part in qs.split("&"):
        if not part:
            continue
        k, _, v = part.partition("=")
        q[unquote(k)] = unquote(v)
    return q


class Request:
    def __init__(self, method, path, query, headers, reader, writer):
        self.method = method
        self.path = path
        self.query = query
        self.headers = headers
        self.reader = reader
        self.writer = writer
        cl = headers.get("content-length")
        try:
            self.length = int(cl) if cl is not None else None
        except ValueError:
            raise HTTPError(400, "bad Content-Length")
        if self.length is not None and self.length < 0:
            raise HTTPError(400, "bad Content-Length")
        self.cors = path.startswith("/calibrate")
        self.responded = False
        self.left = None

    def qint(self, name, default, lo=None, hi=None):
        v = self.query.get(name)
        if v is None or v == "":
            return default
        try:
            v = int(v)
        except ValueError:
            raise HTTPError(400, "%s must be an integer" % name)
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            raise HTTPError(400, "%s out of range" % name)
        return v

    async def read_chunk(self, size=2048):
        """Next piece of the body, or None once it has all been read. Raises
        HTTPError if the client stops before Content-Length bytes."""
        if self.length is None:
            raise HTTPError(411, "Content-Length required")
        if self.left is None:
            self.left = self.length
        if self.left == 0:
            return None
        chunk = await asyncio.wait_for(self.reader.read(min(size, self.left)), CHUNK_TIMEOUT)
        if not chunk:
            raise HTTPError(400, "body truncated: %d of %d bytes missing" % (self.left, self.length))
        self.left -= len(chunk)
        return chunk

    async def body(self, limit):
        if self.length is None:
            raise HTTPError(411, "Content-Length required")
        if self.length > limit:
            raise HTTPError(413, "body larger than %d bytes" % limit)
        buf = bytearray()
        while True:
            chunk = await self.read_chunk(1024)
            if chunk is None:
                return bytes(buf)
            buf.extend(chunk)

    async def json(self, limit=4096):
        try:
            data = json.loads(await self.body(limit))
        except ValueError:
            raise HTTPError(400, "body is not valid JSON")
        if not isinstance(data, dict):
            raise HTTPError(400, "expected a JSON object")
        return data

    async def send(self, code, body, ctype="application/json", extra=""):
        self.responded = True
        if isinstance(body, str):
            body = body.encode()
        head = "HTTP/1.0 %d %s\r\nContent-Type: %s\r\nContent-Length: %d\r\nConnection: close\r\n%s%s\r\n" % (
            code,
            REASONS.get(code, "OK"),
            ctype,
            len(body),
            CORS if self.cors else "",
            extra,
        )
        self.writer.write(head.encode())
        if body:
            self.writer.write(body)
        await asyncio.wait_for(self.writer.drain(), CHUNK_TIMEOUT)


class Server:
    def __init__(self, port, token):
        self.port = port
        self.token = token
        self.routes = []
        self.active = 0
        self.requests = 0
        self.rejected = 0
        self.srv = None
        self.hb = time.ticks_ms()

    def route(self, method, path, handler, auth=False):
        """path ending in "/*" matches any suffix."""
        self.routes.append((method, path, handler, auth))

    def authorized(self, headers):
        if not self.token:
            return False
        given = headers.get("x-tree-token")
        auth = headers.get("authorization", "")
        if auth[:7].lower() == "bearer ":
            given = auth[7:].strip()
        return bool(given) and bootmgr.eq(given.encode(), self.token.encode())

    def _match(self, method, path):
        allowed = False
        for m, p, h, auth in self.routes:
            if p == path or (p.endswith("/*") and path.startswith(p[:-1]) and len(path) > len(p) - 1):
                if m == method or (m == "GET|POST" and method in ("GET", "POST")):
                    return h, auth
                allowed = True
        return None, allowed

    async def start(self):
        self.srv = await asyncio.start_server(self._conn, "0.0.0.0", self.port, backlog=4)
        log("http: listening on port %d" % self.port)

    def alive(self):
        return self.srv is not None and not self.srv.task.done()

    async def _head(self, r):
        line = await r.readline()
        if not line:
            return None
        if len(line) > MAX_LINE:
            raise HTTPError(400, "request line too long")
        parts = line.decode().strip().split(" ")
        if len(parts) < 2:
            raise HTTPError(400, "bad request line")
        method, target = parts[0].upper(), parts[1]
        headers = {}
        for _ in range(MAX_HEADERS + 1):
            h = await r.readline()
            if len(h) > MAX_LINE:
                raise HTTPError(400, "header too long")
            h = h.decode().strip()
            if not h:
                break
            k, _, v = h.partition(":")
            headers[k.strip().lower()] = v.strip()
        else:
            raise HTTPError(400, "too many headers")
        path, _, qs = target.partition("?")
        return method, path, parse_query(qs), headers

    async def _conn(self, r, w):
        self.requests += 1
        req = None
        if self.active >= MAX_CONNECTIONS:
            self.rejected += 1
            try:
                w.write(b"HTTP/1.0 503 Service Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                await asyncio.wait_for(w.drain(), 2)
            except Exception:
                pass
            await self._close(w)
            return
        self.active += 1
        try:
            head = await asyncio.wait_for(self._head(r), HEAD_TIMEOUT)
            if head is None:
                return
            method, path, query, headers = head
            req = Request(method, path, query, headers, r, w)
            if method == "OPTIONS":
                await req.send(204 if req.cors else 405, b"")
                return
            handler, auth = self._match(method, path)
            if handler is None:
                raise HTTPError(405 if auth else 404, "method not allowed" if auth else "not found")
            if auth and not self.authorized(headers):
                if not self.token:
                    raise HTTPError(403, "no HTTP_TOKEN configured in config.py; mutating endpoints are disabled")
                raise HTTPError(401, "missing or bad token")
            res = await handler(req)
            if res is not None and not req.responded:
                code, obj = res
                await req.send(code, json.dumps(obj))
        except HTTPError as e:
            await self._error(req, w, e.code, e.msg)
        except asyncio.TimeoutError:
            await self._error(req, w, 408, "timeout")
        except Exception as e:
            log("http: handler error: %r" % (e,))
            await self._error(req, w, 500, repr(e))
        finally:
            self.active -= 1
            await self._close(w)

    async def _error(self, req, w, code, msg):
        try:
            if req is not None:
                if not req.responded:
                    await req.send(code, json.dumps({"error": msg}))
            else:
                body = json.dumps({"error": msg}).encode()
                w.write(b"HTTP/1.0 %d %s\r\nContent-Type: application/json\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % (code, REASONS.get(code, "Error").encode(), len(body)))
                w.write(body)
                await asyncio.wait_for(w.drain(), 2)
        except Exception:
            pass

    async def _close(self, w):
        try:
            w.close()
            await asyncio.wait_for(w.wait_closed(), 2)
        except Exception:
            pass
