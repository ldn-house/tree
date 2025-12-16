"""
Self-update module for Pi Pico 2W.

Downloads firmware updates from GitHub CI artifacts via nightly.link.
Handles HTTP redirects, ZIP parsing, and file extraction.
"""

import gc
import os
import socket
import struct
import time

try:
    import zlib
    ZLIB_AVAILABLE = True
except ImportError:
    ZLIB_AVAILABLE = False

try:
    from github import GITHUB_BRANCH, nightly_link_url, FIRMWARE_FILES
except ImportError:
    # Fallback if github.py not yet installed
    GITHUB_BRANCH = "main"
    FIRMWARE_FILES = ["main.py", "ota.py", "mqtt.py", "selfupdate.py", "github.py"]
    def nightly_link_url(branch="main"):
        # URL-encode slash in branch name (e.g., "claude/feature" -> "claude%2Ffeature")
        encoded = branch.replace("/", "%2F")
        return f"https://nightly.link/ldn-house/tree/workflows/build.yml/{encoded}/firmware.zip"

# Update state
_update_in_progress = False
_update_status = "idle"
_update_error = None


def get_status():
    """Get current update status."""
    return {
        "in_progress": _update_in_progress,
        "status": _update_status,
        "error": _update_error,
    }


def _parse_url(url):
    """Parse URL into (host, port, path, use_ssl)."""
    if url.startswith("https://"):
        url = url[8:]
        port = 443
        use_ssl = True
    elif url.startswith("http://"):
        url = url[7:]
        port = 80
        use_ssl = False
    else:
        raise ValueError(f"Invalid URL scheme: {url}")

    # Split host and path
    slash_idx = url.find("/")
    if slash_idx == -1:
        host = url
        path = "/"
    else:
        host = url[:slash_idx]
        path = url[slash_idx:]

    # Check for port in host
    if ":" in host:
        host, port_str = host.split(":", 1)
        port = int(port_str)

    return host, port, path, use_ssl


def _http_request(url, max_redirects=5):
    """Make HTTP GET request, following redirects. Returns (status, headers, body)."""
    global _update_status

    for redirect_count in range(max_redirects + 1):
        host, port, path, use_ssl = _parse_url(url)
        _update_status = f"connecting to {host}..."

        # Create socket
        addr = socket.getaddrinfo(host, port)[0][-1]
        sock = socket.socket()
        sock.settimeout(30)

        try:
            sock.connect(addr)

            # Wrap with SSL if needed
            if use_ssl:
                import ssl
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                ctx.verify_mode = ssl.CERT_NONE  # Skip cert verification (limited CA on Pico)
                sock = ctx.wrap_socket(sock, server_hostname=host)

            # Send request
            request = f"GET {path} HTTP/1.0\r\nHost: {host}\r\nUser-Agent: PicoOTA/1.0\r\n\r\n"
            sock.send(request.encode())

            # Read response
            _update_status = "downloading..."
            response = b""
            while True:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                response += chunk
                # Yield to other tasks periodically
                if len(response) % 16384 == 0:
                    gc.collect()

        finally:
            sock.close()

        # Parse response
        header_end = response.find(b"\r\n\r\n")
        if header_end == -1:
            raise RuntimeError("Invalid HTTP response")

        header_data = response[:header_end].decode("utf-8")
        body = response[header_end + 4:]

        # Parse status line
        lines = header_data.split("\r\n")
        status_line = lines[0]
        parts = status_line.split(" ", 2)
        status_code = int(parts[1])

        # Parse headers
        headers = {}
        for line in lines[1:]:
            if ": " in line:
                key, value = line.split(": ", 1)
                headers[key.lower()] = value

        # Handle redirects
        if status_code in (301, 302, 303, 307, 308):
            if "location" not in headers:
                raise RuntimeError(f"Redirect {status_code} without Location header")
            url = headers["location"]
            print(f"Redirecting to: {url[:60]}...")
            continue

        return status_code, headers, body

    raise RuntimeError(f"Too many redirects (>{max_redirects})")


def _parse_zip(data):
    """Parse ZIP file and yield (filename, content) tuples."""
    pos = 0
    while pos < len(data) - 4:
        # Check for local file header signature (PK\x03\x04)
        sig = data[pos:pos+4]
        if sig != b"PK\x03\x04":
            break  # End of local files (probably central directory)

        # Parse local file header (30 bytes minimum)
        # Offset 8: compression method (2 bytes)
        # Offset 18: compressed size (4 bytes)
        # Offset 22: uncompressed size (4 bytes)
        # Offset 26: filename length (2 bytes)
        # Offset 28: extra field length (2 bytes)
        compression = struct.unpack_from("<H", data, pos + 8)[0]
        compressed_size = struct.unpack_from("<I", data, pos + 18)[0]
        uncompressed_size = struct.unpack_from("<I", data, pos + 22)[0]
        filename_len = struct.unpack_from("<H", data, pos + 26)[0]
        extra_len = struct.unpack_from("<H", data, pos + 28)[0]

        # Read filename
        filename_start = pos + 30
        filename = data[filename_start:filename_start + filename_len].decode("utf-8")

        # Read file data
        data_start = filename_start + filename_len + extra_len
        compressed_data = data[data_start:data_start + compressed_size]

        # Decompress if needed
        if compression == 0:  # Stored (no compression)
            content = compressed_data
        elif compression == 8:  # Deflate
            if not ZLIB_AVAILABLE:
                raise RuntimeError(f"zlib not available, cannot decompress {filename}")
            # zlib.decompress with -15 wbits for raw deflate
            content = zlib.decompress(compressed_data, -15)
        else:
            raise RuntimeError(f"Unsupported compression method {compression} for {filename}")

        yield filename, content

        # Move to next file
        pos = data_start + compressed_size


def check_for_updates(branch=GITHUB_BRANCH):
    """Check if updates are available from GitHub CI.

    Returns dict with update info or None if check fails.
    Note: nightly.link doesn't provide version info, so we always report
    updates as available. The actual files may or may not be different.
    """
    global _update_status, _update_error

    try:
        _update_status = "checking..."
        url = nightly_link_url(branch)

        # Just do a HEAD-like request by downloading
        # nightly.link doesn't support HEAD, so we have to download
        # For now, just report that updates are available
        return {
            "available": True,
            "branch": branch,
            "url": url,
            "note": "Download to check actual changes",
        }

    except Exception as e:
        _update_error = str(e)
        _update_status = f"check failed: {e}"
        return None


def download_and_apply(branch=GITHUB_BRANCH, reboot=True):
    """Download and apply firmware update from GitHub CI.

    Returns True on success, False on failure.
    If reboot=True, reboots after successful update.
    """
    global _update_in_progress, _update_status, _update_error
    import machine

    if _update_in_progress:
        _update_error = "Update already in progress"
        return False

    _update_in_progress = True
    _update_error = None
    files_updated = []

    try:
        # Build URL and download
        url = nightly_link_url(branch)
        _update_status = "downloading from GitHub CI..."
        print(f"Downloading: {url}")

        gc.collect()
        status, headers, body = _http_request(url)

        if status != 200:
            raise RuntimeError(f"HTTP {status}")

        print(f"Downloaded {len(body)} bytes")
        _update_status = "extracting..."

        # Parse ZIP and extract files
        gc.collect()
        for filename, content in _parse_zip(body):
            # Skip directories and non-firmware files
            if filename.endswith("/"):
                continue

            # Use basename (ZIP might have directory prefix)
            basename = filename.split("/")[-1]

            # Only update known firmware files
            if basename not in FIRMWARE_FILES:
                print(f"Skipping unknown file: {basename}")
                continue

            print(f"Updating {basename} ({len(content)} bytes)...")
            _update_status = f"writing {basename}..."

            # Write to temp file then rename (atomic-ish)
            temp_name = "_update_tmp"
            try:
                with open(temp_name, "wb") as f:
                    f.write(content)

                # Remove old file and rename
                try:
                    os.remove(basename)
                except OSError:
                    pass
                os.rename(temp_name, basename)

                files_updated.append(basename)
            except Exception as e:
                print(f"Failed to write {basename}: {e}")
                try:
                    os.remove(temp_name)
                except OSError:
                    pass
                raise

        # Clean up
        del body
        gc.collect()

        if not files_updated:
            _update_status = "no files updated"
            _update_in_progress = False
            return False

        _update_status = f"updated {len(files_updated)} files"
        print(f"Updated: {', '.join(files_updated)}")

        if reboot:
            _update_status = "rebooting..."
            print("Rebooting in 1 second...")
            time.sleep(1)
            machine.reset()

        _update_in_progress = False
        return True

    except Exception as e:
        _update_error = str(e)
        _update_status = f"failed: {e}"
        print(f"Update failed: {e}")
        _update_in_progress = False
        return False
