"""Bounded stream reading and deterministic socket cleanup.

MicroPython's Stream.readline() keeps appending until it sees a newline, so
a peer that never sends one can fill the heap before any length check runs.
BufReader reads in small pieces and fails as soon as a line, or the whole
header block, would exceed its budget. Body reads go through the same
buffer, so nothing read ahead is lost.

Stream.close() is a no-op in MicroPython asyncio; the socket only closes in
wait_closed(). close_now() closes it immediately, so a dropped MQTT session
or a finished HTTP fetch never keeps a socket (or TLS buffers) alive.
"""

import asyncio

CHUNK = 256


class TooLong(Exception):
    pass


class BufReader:
    def __init__(self, stream, head_budget=4096):
        self.s = stream
        self.buf = bytearray()
        self.head_left = head_budget

    async def readline(self, limit=1024):
        """One line including b"\\n" (b"" at EOF). Raises TooLong."""
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[: i + 1])
                self.buf = self.buf[i + 1 :]
                self.head_left -= len(line)
                if len(line) > limit or self.head_left < 0:
                    raise TooLong()
                return line
            if len(self.buf) >= limit or len(self.buf) >= self.head_left:
                raise TooLong()
            chunk = await self.s.read(min(CHUNK, limit - len(self.buf) + 1))
            if not chunk:
                line = bytes(self.buf)
                self.buf = bytearray()
                return line
            self.buf.extend(chunk)

    async def readline_capped(self, limit=1024):
        """One line, for a reader that only needs some of them. Returns
        (line, True), or (its first limit bytes, False) when it is longer:
        the rest is read and dropped, so no more than about limit bytes are
        held however long the line is. Every byte still counts against the
        head budget (TooLong once it is used up). (b"", True) at EOF."""
        head = None  # set once the line has outgrown limit
        while True:
            i = self.buf.find(b"\n")
            if i >= 0:
                line = bytes(self.buf[: i + 1])
                self.buf = self.buf[i + 1 :]
                self.head_left -= len(line)
                if self.head_left < 0:
                    raise TooLong()
                if head is None and len(line) <= limit:
                    return line, True
                return (line[:limit] if head is None else head), False
            if head is None and len(self.buf) > limit:
                head = bytes(self.buf[:limit])
            if head is not None:
                self.head_left -= len(self.buf)
                self.buf = bytearray()
            if len(self.buf) >= self.head_left:
                raise TooLong()
            chunk = await self.s.read(CHUNK)
            if not chunk:
                line = bytes(self.buf)
                self.buf = bytearray()
                return (line, True) if head is None else (head, False)
            self.buf.extend(chunk)

    async def read(self, n):
        if self.buf:
            out = bytes(self.buf[:n])
            self.buf = self.buf[n:]
            return out
        return await self.s.read(n)

    def close(self):
        close_now(self.s)


def close_now(stream):
    """Close an asyncio stream's socket right away (see module docstring)."""
    if stream is None:
        return
    try:
        stream.close()
    except Exception:
        pass
    sock = getattr(stream, "s", None)
    if sock is not None:
        try:
            sock.close()
        except Exception:
            pass


async def aclose(stream, timeout=2):
    close_now(stream)
    try:
        await asyncio.wait_for(stream.wait_closed(), timeout)
    except Exception:
        pass
