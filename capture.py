#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "opencv-python",
#     "requests",
#     "numpy",
# ]
# ///
"""
LED Capture Script - Captures images of each LED for 3D position mapping.

Usage:
    # Sequential mode (one LED at a time) - most accurate
    ./capture.py --host 192.168.2.149 --angle front

    # Binary mode (faster, 9 frames per angle)
    ./capture.py --host 192.168.2.149 --angle front --binary

    # Preview camera without capturing
    ./capture.py --preview

    # Capture specific LED range
    ./capture.py --host 192.168.2.149 --angle front --start 0 --end 100
"""

import argparse
import cv2
import json
import numpy as np
import os
import requests
import sys
import time
from pathlib import Path

LED_COUNT = 500
CAPTURE_DIR = Path("captures")


def get_pico_url(host: str, port: int = 8080) -> str:
    return f"http://{host}:{port}"


def pico_request(base_url: str, endpoint: str, timeout: float = 5.0) -> dict | None:
    """Make request to Pico and return JSON response."""
    try:
        resp = requests.get(f"{base_url}{endpoint}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None


def start_calibration(base_url: str) -> bool:
    """Enter calibration mode on Pico."""
    result = pico_request(base_url, "/calibrate/start")
    if result and result.get("success"):
        print("Entered calibration mode")
        return True
    print("Failed to enter calibration mode")
    return False


def stop_calibration(base_url: str) -> bool:
    """Exit calibration mode on Pico."""
    result = pico_request(base_url, "/calibrate/stop")
    if result and result.get("success"):
        print("Exited calibration mode")
        return True
    return False


def set_led(base_url: str, led_index: int) -> bool:
    """Turn on specific LED (or -1 for all off)."""
    result = pico_request(base_url, f"/calibrate/led?n={led_index}")
    return result is not None and result.get("success", False)


def set_binary_pattern(base_url: str, bit: int) -> bool:
    """Set binary encoding pattern for given bit."""
    result = pico_request(base_url, f"/calibrate/binary?bit={bit}")
    return result is not None and result.get("success", False)


def init_camera(camera_index: int = 0, focus: int = -1) -> cv2.VideoCapture:
    """Initialize camera with optimal settings for LED capture.

    Args:
        camera_index: Which camera to use
        focus: Fixed focus value 0-250 (0=far, 250=near), or -1 for autofocus
    """
    # Try AVFoundation backend on macOS for better control
    cap = cv2.VideoCapture(camera_index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        # Fall back to default
        cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}")

    # Set resolution - C920 supports 1080p
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    # For Logitech C920: Focus range is 0-250, autofocus needs to be explicitly disabled
    # IMPORTANT: Set autofocus BEFORE setting focus value
    if focus >= 0:
        # Disable autofocus first (critical for C920)
        result = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        print(f"Autofocus disable: {'OK' if result else 'FAILED'}")
        time.sleep(0.3)  # Give camera time to switch modes

        # Now set focus (C920 uses 0-250 range)
        result = cap.set(cv2.CAP_PROP_FOCUS, focus)
        print(f"Focus set to {focus}: {'OK' if result else 'FAILED'}")
    else:
        # Enable autofocus
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)

    # Let camera warm up and stabilize
    print("Waiting for camera to stabilize...")
    for _ in range(30):
        cap.read()
        time.sleep(0.05)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    current_focus = cap.get(cv2.CAP_PROP_FOCUS)
    autofocus_state = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    print(f"Camera: {actual_width}x{actual_height}, focus={current_focus}, autofocus={autofocus_state}")

    return cap


def rotate_frame(frame: np.ndarray, rotation: str) -> np.ndarray:
    """Rotate frame to portrait orientation."""
    if rotation == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif rotation == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def capture_frame(cap: cv2.VideoCapture, settle_frames: int = 3) -> np.ndarray:
    """Capture a frame after letting the image settle."""
    # Discard frames to let exposure settle
    for _ in range(settle_frames):
        cap.read()

    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Failed to capture frame")
    return frame


def preview_camera(camera_index: int = 0, rotation: str = "none", initial_focus: int = -1):
    """Preview camera feed without capturing."""
    cap = init_camera(camera_index, initial_focus)

    # Focus control state
    focus = initial_focus if initial_focus >= 0 else 50  # Default to mid-range
    autofocus = initial_focus < 0  # Start with autofocus if no initial value

    print("Controls:")
    print("  'q'/ESC = quit, 's' = save, 'r' = rotate")
    print("  'a' = toggle autofocus")
    print("  '+'/'-' or UP/DOWN = adjust focus (when autofocus off)")
    print("  '['/']' = fine adjust focus")
    print("(Make sure the preview window is focused)")

    rotations = ["none", "cw", "ccw"]
    rot_idx = rotations.index(rotation) if rotation in rotations else 0
    window_name = "Camera Preview"

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Apply rotation
        display = rotate_frame(frame, rotations[rot_idx])

        # Show frame with info overlay
        h, w = display.shape[:2]
        focus_str = "AUTO" if autofocus else f"{focus}"

        # Large focus display at top
        focus_color = (0, 255, 255) if not autofocus else (0, 255, 0)  # Yellow when manual
        cv2.putText(display, f"FOCUS: {focus_str}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, focus_color, 3)

        # Instructions at bottom
        cv2.putText(display, f"'a'=autofocus '+'/'-'=adjust '['/']'=fine | rot={rotations[rot_idx]}",
                    (10, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(display, f"'s'=save 'q'=quit | {w}x{h}",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow(window_name, display)

        # Check for key press
        key = cv2.waitKey(30) & 0xFF

        # Check if window was closed
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if key == ord('q') or key == 27:  # 'q' or ESC
            break
        elif key == ord('r'):
            rot_idx = (rot_idx + 1) % len(rotations)
            print(f"Rotation: {rotations[rot_idx]}")
        elif key == ord('s'):
            CAPTURE_DIR.mkdir(exist_ok=True)
            path = CAPTURE_DIR / "preview.jpg"
            cv2.imwrite(str(path), display)
            print(f"Saved: {path} (focus={focus_str})")
        elif key == ord('a'):
            autofocus = not autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
            time.sleep(0.2)  # Let camera switch modes
            if not autofocus:
                cap.set(cv2.CAP_PROP_FOCUS, focus)
            print(f"Autofocus: {'ON' if autofocus else 'OFF'}")
        elif key in [ord('+'), ord('='), 82]:  # + or up arrow
            if not autofocus:
                focus = min(250, focus + 10)  # C920 max is 250
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                print(f"Focus: {focus}")
        elif key in [ord('-'), 84]:  # - or down arrow
            if not autofocus:
                focus = max(0, focus - 10)
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                print(f"Focus: {focus}")
        elif key == ord(']'):  # Fine adjust up
            if not autofocus:
                focus = min(250, focus + 5)
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                print(f"Focus: {focus}")
        elif key == ord('['):  # Fine adjust down
            if not autofocus:
                focus = max(0, focus - 5)
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                print(f"Focus: {focus}")

    cap.release()
    cv2.destroyAllWindows()

    if not autofocus:
        print(f"\nTo use this focus setting for capture, run:")
        print(f"  ./capture.py --host <ip> --angle front --focus {focus}")


def capture_sequential(base_url: str, angle: str, cap: cv2.VideoCapture,
                       start: int = 0, end: int = LED_COUNT, settle_ms: int = 100,
                       rotation: str = "none"):
    """Capture one image per LED (sequential mode)."""
    angle_dir = CAPTURE_DIR / angle / "sequential"
    angle_dir.mkdir(parents=True, exist_ok=True)

    # Enter calibration mode first
    result = pico_request(base_url, "/calibrate/start")
    print(f"Calibration start: {result}")
    time.sleep(0.3)  # Let Pico process the mode change

    # Pre-focus: light several LEDs to give autofocus something to lock onto
    print("\n=== PRE-FOCUS STEP ===")
    print("Lighting LEDs for autofocus to lock onto...")
    result = pico_request(base_url, "/calibrate/binary?bit=3")  # Lights ~250 LEDs spread out
    print(f"Binary pattern: {result}")
    time.sleep(0.5)  # Wait for LEDs to actually light up

    # Show preview so user can verify focus
    print("Check the preview window - is the image in focus?")
    print("Press SPACE when focus looks good, or 'q' to abort")

    window_name = "Focus Check - SPACE to continue, Q to abort"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = rotate_frame(frame, rotation)
        h, w = display.shape[:2]
        cv2.putText(display, "Is this in focus? SPACE=yes, Q=abort",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(100) & 0xFF
        if key == ord(' '):
            print("Focus confirmed! Starting capture...")
            break
        elif key == ord('q') or key == 27:
            print("Aborted by user")
            cv2.destroyAllWindows()
            set_led(base_url, -1)
            return

    cv2.destroyWindow(window_name)

    # Capture background (all LEDs off)
    print("Capturing background...")
    set_led(base_url, -1)
    time.sleep(0.2)
    bg_frame = rotate_frame(capture_frame(cap), rotation)
    cv2.imwrite(str(angle_dir / "background.jpg"), bg_frame)

    # Capture each LED
    total = end - start
    for i, led in enumerate(range(start, end)):
        set_led(base_url, led)
        time.sleep(settle_ms / 1000)

        frame = rotate_frame(capture_frame(cap, settle_frames=2), rotation)
        filename = angle_dir / f"led_{led:04d}.jpg"
        cv2.imwrite(str(filename), frame)

        # Progress
        if (i + 1) % 10 == 0 or i == total - 1:
            pct = (i + 1) / total * 100
            print(f"Progress: {i + 1}/{total} ({pct:.1f}%) - LED {led}")

        # Show live preview
        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Capture interrupted")
            break

    # Turn off LEDs and exit calibration mode
    set_led(base_url, -1)
    pico_request(base_url, "/calibrate/stop")
    print(f"Captured {end - start} frames to {angle_dir}")


def capture_binary(base_url: str, angle: str, cap: cv2.VideoCapture, settle_ms: int = 500,
                   rotation: str = "none"):
    """Capture binary encoding frames (9 bits for 500 LEDs)."""
    angle_dir = CAPTURE_DIR / angle / "binary"
    angle_dir.mkdir(parents=True, exist_ok=True)

    # Enter calibration mode first
    result = pico_request(base_url, "/calibrate/start")
    print(f"Calibration start: {result}")
    time.sleep(0.5)  # Let Pico process the mode change

    # Need ceil(log2(500)) = 9 bits
    num_bits = 9

    # Pre-focus: light several LEDs to give autofocus something to lock onto
    print("\n=== PRE-FOCUS STEP ===")
    print("Lighting LEDs for autofocus to lock onto...")
    result = pico_request(base_url, "/calibrate/binary?bit=3")  # Lights ~250 LEDs spread out
    print(f"Binary pattern result: {result}")
    time.sleep(1.5)  # Give LEDs time to stabilize

    # Show preview so user can verify focus
    print("Check the preview window - is the image in focus?")
    print("Press SPACE when focus looks good, or 'q' to abort")

    window_name = "Focus Check - SPACE to continue, Q to abort"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = rotate_frame(frame, rotation)
        cv2.putText(display, "Is this in focus? SPACE=yes, Q=abort",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.imshow(window_name, display)

        key = cv2.waitKey(100) & 0xFF
        if key == ord(' '):
            print("Focus confirmed! Starting capture...")
            break
        elif key == ord('q') or key == 27:
            print("Aborted by user")
            cv2.destroyAllWindows()
            set_led(base_url, -1)
            pico_request(base_url, "/calibrate/stop")
            return

    cv2.destroyWindow(window_name)

    # Capture background (all LEDs off)
    print("Capturing background...")
    set_led(base_url, -1)
    time.sleep(0.3)
    bg_frame = rotate_frame(capture_frame(cap, settle_frames=5), rotation)
    cv2.imwrite(str(angle_dir / "background.jpg"), bg_frame)

    # Capture each bit pattern
    for bit in range(num_bits):
        print(f"Capturing bit {bit} pattern...")
        result = set_binary_pattern(base_url, bit)
        print(f"  Binary pattern {bit}: {result}")
        time.sleep(settle_ms / 1000)  # Wait for LEDs to update

        frame = rotate_frame(capture_frame(cap, settle_frames=3), rotation)
        filename = angle_dir / f"bit_{bit}.jpg"
        cv2.imwrite(str(filename), frame)

        # Show preview
        cv2.imshow("Capture", frame)
        cv2.waitKey(100)

    # Turn off LEDs and exit calibration mode
    set_led(base_url, -1)
    pico_request(base_url, "/calibrate/stop")
    print(f"Captured {num_bits} binary frames to {angle_dir}")


def save_capture_metadata(angle: str, mode: str, host: str, start: int, end: int,
                          rotation: str = "none"):
    """Save metadata about this capture session."""
    metadata = {
        "angle": angle,
        "mode": mode,
        "host": host,
        "led_count": LED_COUNT,
        "range": {"start": start, "end": end},
        "rotation": rotation,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    meta_path = CAPTURE_DIR / angle / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Capture LED images for 3D mapping")
    parser.add_argument("--host", default="192.168.2.149", help="Pico IP address")
    parser.add_argument("--port", type=int, default=8080, help="Pico OTA port")
    parser.add_argument("--angle", default="front", help="Angle name (e.g., front, side, back)")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--binary", action="store_true", help="Use binary encoding mode")
    parser.add_argument("--preview", action="store_true", help="Preview camera only")
    parser.add_argument("--start", type=int, default=0, help="Start LED index")
    parser.add_argument("--end", type=int, default=LED_COUNT, help="End LED index")
    parser.add_argument("--settle", type=int, default=150, help="Settle time in ms between captures")
    parser.add_argument("--rotate", choices=["none", "cw", "ccw"], default="cw",
                        help="Rotate camera to portrait (cw=clockwise, ccw=counter-clockwise, default=cw)")
    parser.add_argument("--focus", type=int, default=-1,
                        help="Fixed focus value 0-255 (0=far, 255=near). -1=autofocus (default)")

    args = parser.parse_args()

    if args.preview:
        preview_camera(args.camera, args.rotate, args.focus)
        return

    base_url = get_pico_url(args.host, args.port)

    # Test connection
    print(f"Connecting to Pico at {base_url}...")
    status = pico_request(base_url, "/")
    if not status:
        print("Failed to connect to Pico")
        sys.exit(1)
    print(f"Connected: {status.get('hostname', 'unknown')}")

    # Initialize camera
    print("Initializing camera...")
    cap = init_camera(args.camera, args.focus)

    # Enter calibration mode
    if not start_calibration(base_url):
        cap.release()
        sys.exit(1)

    try:
        CAPTURE_DIR.mkdir(exist_ok=True)

        if args.binary:
            capture_binary(base_url, args.angle, cap, args.settle, args.rotate)
            save_capture_metadata(args.angle, "binary", args.host, 0, LED_COUNT, args.rotate)
        else:
            capture_sequential(base_url, args.angle, cap, args.start, args.end, args.settle, args.rotate)
            save_capture_metadata(args.angle, "sequential", args.host, args.start, args.end, args.rotate)

    finally:
        # Always exit calibration mode and cleanup
        stop_calibration(base_url)
        cap.release()
        cv2.destroyAllWindows()

    print("\nCapture complete!")
    print(f"Images saved to: {CAPTURE_DIR / args.angle}")
    print("\nNext steps:")
    print(f"  1. Move camera to a different angle (e.g., 'side')")
    print(f"  2. Run: ./capture.py --host {args.host} --angle side")
    print(f"  3. Then run: ./analyze.py to extract LED positions")


if __name__ == "__main__":
    main()
