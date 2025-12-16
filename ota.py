"""
OTA Update Server with mDNS for Pi Pico 2W.
Provides HTTP endpoints for code updates and status checks.
"""

import network
import socket
import struct
import time
import machine
import gc
import _thread

try:
    from config import WIFI_SSID, WIFI_PASSWORD, HOSTNAME, OTA_PORT
except ImportError:
    print("ERROR: config.py not found. Create it with WiFi credentials.")
    raise

# Global state
_server_running = False
_mdns_running = False
_start_time = None
_ip_address = None

# Animation state (set by main.py)
current_animation = None
animation_index = 0
total_animations = 0

# Calibration state (checked by main.py, set by HTTP endpoints)
calibration_mode = False
calibration_led = -1  # -1 = all off, -2 = binary mode, 0+ = specific LED
calibration_binary_bit = 0

# These will be set by main.py after import
strip = None
LED_COUNT = 500

# Note: Calibration LED writing is handled by main.py's update_calibration_leds()
# to ensure thread safety (NeoPixel PIO isn't thread-safe)


# mDNS constants
MDNS_ADDR = "224.0.0.251"
MDNS_PORT = 5353


_last_wifi_check = 0
_wifi_check_interval = 10000  # Check every 10 seconds

def connect_wifi(timeout_s=30):
    """Connect to WiFi and return IP address."""
    global _ip_address

    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)

    # Set hostname before connecting (helps with some routers)
    try:
        wlan.config(hostname=HOSTNAME)
    except Exception:
        pass  # Not all firmware versions support this

    if wlan.isconnected():
        _ip_address = wlan.ifconfig()[0]
        print(f"Already connected: {_ip_address}")
        return _ip_address

    print(f"Connecting to {WIFI_SSID}...")
    wlan.connect(WIFI_SSID, WIFI_PASSWORD)

    start = time.time()
    while not wlan.isconnected():
        if time.time() - start > timeout_s:
            raise RuntimeError("WiFi connection timeout")
        time.sleep(0.5)

    _ip_address = wlan.ifconfig()[0]
    print(f"Connected: {_ip_address}")
    print(f"Hostname: {HOSTNAME}.local")
    return _ip_address


def check_wifi():
    """Check WiFi connection and reconnect if needed. Call periodically from main loop."""
    global _last_wifi_check, _ip_address

    now = time.ticks_ms()
    if time.ticks_diff(now, _last_wifi_check) < _wifi_check_interval:
        return True
    _last_wifi_check = now

    wlan = network.WLAN(network.STA_IF)
    if wlan.isconnected():
        return True

    print("WiFi disconnected, reconnecting...")
    try:
        wlan.active(False)
        time.sleep(1)
        wlan.active(True)
        wlan.connect(WIFI_SSID, WIFI_PASSWORD)

        for _ in range(20):  # 10 second timeout
            if wlan.isconnected():
                _ip_address = wlan.ifconfig()[0]
                print(f"WiFi reconnected: {_ip_address}")
                return True
            time.sleep(0.5)

        print("WiFi reconnection failed")
        return False
    except Exception as e:
        print(f"WiFi reconnect error: {e}")
        return False


def _build_mdns_response(query_id, hostname, ip):
    """Build mDNS response packet for A record query."""
    # Header: ID, Flags (response, authoritative), QD=0, AN=1, NS=0, AR=0
    header = struct.pack(">HHHHHH", query_id, 0x8400, 0, 1, 0, 0)

    # Answer: name, type A, class IN (cache flush), TTL, rdlength, IP
    name_parts = [hostname.encode(), b"local"]
    name = b"".join(bytes([len(p)]) + p for p in name_parts) + b"\x00"

    ip_bytes = bytes(int(x) for x in ip.split("."))
    answer = name + struct.pack(">HHIH", 1, 0x8001, 120, 4) + ip_bytes

    return header + answer


def _parse_mdns_query(data):
    """Parse mDNS query, return (query_id, hostname) or (None, None)."""
    if len(data) < 12:
        return None, None

    query_id = struct.unpack(">H", data[0:2])[0]
    flags = struct.unpack(">H", data[2:4])[0]

    # Skip responses (QR bit set)
    if flags & 0x8000:
        return None, None

    # Parse question section
    pos = 12
    labels = []
    while pos < len(data) and data[pos] != 0:
        length = data[pos]
        pos += 1
        if pos + length > len(data):
            break
        labels.append(data[pos:pos + length].decode())
        pos += length

    if len(labels) >= 2 and labels[-1] == "local":
        return query_id, labels[0]

    return None, None


def _mdns_thread():
    """Run mDNS responder in background thread."""
    global _mdns_running, _ip_address

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind(("", MDNS_PORT))
        except OSError as e:
            print(f"mDNS port {MDNS_PORT} busy, skipping mDNS")
            _mdns_running = False
            return

        # Join multicast group - MicroPython compatible
        # IP_ADD_MEMBERSHIP = 35 on most systems
        mcast_addr = bytes([224, 0, 0, 251]) + bytes([0, 0, 0, 0])
        sock.setsockopt(0, 35, mcast_addr)  # IPPROTO_IP=0, IP_ADD_MEMBERSHIP=35
        sock.settimeout(1.0)

        _mdns_running = True
        print(f"mDNS responder started for {HOSTNAME}.local")

        while _mdns_running:
            try:
                data, addr = sock.recvfrom(512)
                query_id, hostname = _parse_mdns_query(data)

                if hostname and hostname.lower() == HOSTNAME.lower():
                    response = _build_mdns_response(query_id, HOSTNAME, _ip_address)
                    sock.sendto(response, (MDNS_ADDR, MDNS_PORT))
            except OSError:
                pass  # Timeout
            except Exception as e:
                print(f"mDNS recv error: {e}")
    except Exception as e:
        print(f"mDNS setup failed: {e}")
        _mdns_running = False


def start_mdns():
    """Start mDNS responder in background thread."""
    global _mdns_running

    if _mdns_running:
        return

    _thread.start_new_thread(_mdns_thread, ())
    time.sleep(0.2)


def _parse_request(client):
    """Parse HTTP request, return (method, path, query_params, content_length, body)."""
    request = b""
    content_length = 0

    # Read headers
    while True:
        chunk = client.recv(1024)
        if not chunk:
            break
        request += chunk
        if b"\r\n\r\n" in request:
            break

    if not request:
        return None, None, {}, 0, b""

    # Split headers and partial body
    header_end = request.find(b"\r\n\r\n")
    headers = request[:header_end].decode("utf-8")
    body = request[header_end + 4:]

    # Parse first line
    lines = headers.split("\r\n")
    first_line = lines[0].split(" ")
    method = first_line[0] if len(first_line) > 0 else "GET"
    full_path = first_line[1] if len(first_line) > 1 else "/"

    # Parse query string
    query_params = {}
    if "?" in full_path:
        path, query_string = full_path.split("?", 1)
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                query_params[key] = value
    else:
        path = full_path

    # Find Content-Length
    for line in lines[1:]:
        if line.lower().startswith("content-length:"):
            content_length = int(line.split(":")[1].strip())
            break

    # Read remaining body if needed
    while len(body) < content_length:
        chunk = client.recv(min(1024, content_length - len(body)))
        if not chunk:
            break
        body += chunk

    return method, path, query_params, content_length, body


def _send_response(client, status, content_type, body):
    """Send HTTP response."""
    response = f"HTTP/1.1 {status}\r\n"
    response += f"Content-Type: {content_type}\r\n"
    response += f"Content-Length: {len(body)}\r\n"
    response += "Connection: close\r\n"
    response += "\r\n"
    client.send(response.encode("utf-8"))
    if isinstance(body, str):
        body = body.encode("utf-8")
    client.send(body)


def _get_status():
    """Return status info as string."""
    global _start_time, _ip_address

    gc.collect()
    free_mem = gc.mem_free()
    uptime = time.time() - _start_time if _start_time else 0

    anim_name = current_animation or "unknown"

    # Check MQTT status
    mqtt_status = "unavailable"
    try:
        import mqtt
        if mqtt.connected:
            mqtt_status = "connected"
        elif mqtt.MQTTClient is None:
            mqtt_status = "umqtt not installed"
        else:
            mqtt_status = "disconnected"
    except ImportError:
        mqtt_status = "not loaded"

    return f"""{{
  "hostname": "{HOSTNAME}",
  "ip": "{_ip_address}",
  "uptime_s": {uptime},
  "free_memory": {free_mem},
  "ota_port": {OTA_PORT},
  "animation": "{anim_name}",
  "animation_index": {animation_index},
  "total_animations": {total_animations},
  "mqtt": "{mqtt_status}"
}}"""


def _handle_update(body, filename="main.py", reboot=True):
    """Write new code to specified file and optionally reboot."""
    if not body:
        return False, "No content received"

    # Basic security: only allow safe file extensions in root
    allowed_extensions = (".py", ".txt", ".json", ".csv")
    if "/" in filename or not any(filename.endswith(ext) for ext in allowed_extensions):
        return False, f"Invalid filename (must be {'/'.join(allowed_extensions)} in root)"

    try:
        # Write to temp file first
        temp_name = "_ota_tmp"
        with open(temp_name, "wb") as f:
            f.write(body)

        # Rename to target (atomic on most filesystems)
        import os
        try:
            os.remove(filename)
        except OSError:
            pass
        os.rename(temp_name, filename)

        if reboot:
            return True, f"Updated {filename} ({len(body)} bytes). Rebooting..."
        else:
            return True, f"Updated {filename} ({len(body)} bytes)."
    except Exception as e:
        return False, f"Update failed: {e}"


def _handle_client(client, addr):
    """Handle a single client connection."""
    try:
        method, path, query_params, content_length, body = _parse_request(client)

        if method is None:
            return

        print(f"{method} {path} from {addr[0]}")

        if path == "/" or path == "/status":
            _send_response(client, "200 OK", "application/json", _get_status())

        elif path == "/update" and method == "POST":
            filename = query_params.get("file", "main.py")
            # Default: reboot for .py files, no reboot for data files
            should_reboot = filename.endswith(".py")
            if "reboot" in query_params:
                should_reboot = query_params["reboot"].lower() in ("1", "true", "yes")

            success, message = _handle_update(body, filename, reboot=should_reboot)
            if success:
                response = f'{{"success": true, "message": "{message}"}}'
                _send_response(client, "200 OK", "application/json", response)
                if should_reboot:
                    client.close()
                    time.sleep(1)
                    machine.reset()
            else:
                response = f'{{"success": false, "error": "{message}"}}'
                _send_response(client, "400 Bad Request", "application/json", response)

        elif path == "/gc":
            gc.collect()
            _send_response(client, "200 OK", "application/json",
                          f'{{"free_memory": {gc.mem_free()}}}')

        elif path == "/mqtt":
            # MQTT debug endpoint
            try:
                import mqtt
                mqtt.check_messages()  # Check for new messages
                info = {
                    "connected": mqtt.connected,
                    "state": mqtt.state,
                    "pending_commands": len(mqtt.pending_commands),
                    "effects": mqtt.effects_list[:3],  # First 3
                }
                _send_response(client, "200 OK", "application/json", str(info).replace("'", '"'))
            except Exception as e:
                _send_response(client, "200 OK", "application/json", f'{{"error": "{e}"}}')

        elif path == "/calibrate/start":
            # Enter calibration mode
            global calibration_mode, calibration_led
            calibration_mode = True
            calibration_led = -1
            print("Entered calibration mode")
            _send_response(client, "200 OK", "application/json",
                          '{"success": true, "mode": "calibration"}')

        elif path == "/calibrate/stop":
            # Exit calibration mode
            global calibration_mode, calibration_led
            calibration_mode = False
            calibration_led = -1
            print("Exited calibration mode")
            _send_response(client, "200 OK", "application/json",
                          '{"success": true, "mode": "normal"}')

        elif path == "/calibrate/led":
            # Set specific LED on (n=index) or all off (n=-1)
            global calibration_led
            if not calibration_mode:
                _send_response(client, "400 Bad Request", "application/json",
                              '{"error": "Not in calibration mode. Call /calibrate/start first"}')
            else:
                calibration_led = int(query_params.get("n", -1))
                _send_response(client, "200 OK", "application/json",
                              f'{{"success": true, "led": {calibration_led}}}')

        elif path == "/calibrate/binary":
            # Set binary pattern (bit=N means LEDs with bit N set are ON)
            global calibration_led, calibration_binary_bit
            if not calibration_mode:
                _send_response(client, "400 Bad Request", "application/json",
                              '{"error": "Not in calibration mode. Call /calibrate/start first"}')
            else:
                calibration_binary_bit = int(query_params.get("bit", 0))
                calibration_led = -2  # Special marker for binary mode
                on_count = sum(1 for i in range(LED_COUNT) if (i >> calibration_binary_bit) & 1)
                print(f"Set binary bit {calibration_binary_bit}, {on_count} LEDs")
                _send_response(client, "200 OK", "application/json",
                              f'{{"success": true, "bit": {calibration_binary_bit}, "leds_on": {on_count}}}')

        elif path == "/calibrate/status":
            # Get calibration status
            _send_response(client, "200 OK", "application/json",
                          f'{{"calibration_mode": {"true" if calibration_mode else "false"}, "current_led": {calibration_led}, "total_leds": 500}}')

        else:
            _send_response(client, "404 Not Found", "text/plain", "Not Found")

    except Exception as e:
        print(f"Client error: {e}")
        try:
            _send_response(client, "500 Internal Server Error", "text/plain", str(e))
        except:
            pass
    finally:
        try:
            client.close()
        except:
            pass


def _server_thread():
    """Run HTTP server in background thread."""
    global _server_running, _start_time

    addr = socket.getaddrinfo("0.0.0.0", OTA_PORT)[0][-1]
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(addr)
    server.listen(1)
    server.settimeout(1.0)  # Allow periodic checks

    _start_time = time.time()
    _server_running = True
    print(f"OTA server listening on port {OTA_PORT}")

    while _server_running:
        try:
            client, addr = server.accept()
            _handle_client(client, addr)
        except OSError:
            pass  # Timeout, continue loop
        gc.collect()


def start_server():
    """Start OTA server and mDNS responder in background threads."""
    global _server_running

    if _server_running:
        print("OTA server already running")
        return

    start_mdns()
    _thread.start_new_thread(_server_thread, ())
    time.sleep(0.5)  # Let server initialize
    print(f"OTA ready: http://{HOSTNAME}.local:{OTA_PORT}/")


def stop_server():
    """Stop OTA server and mDNS responder."""
    global _server_running, _mdns_running
    _server_running = False
    _mdns_running = False
