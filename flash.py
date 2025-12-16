#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpremote", "httpx"]
# ///
"""Flash main.py to Pico via USB or OTA (WiFi)."""

import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

# GitHub repo for fetching CI artifacts via nightly.link
GITHUB_REPO = "ldn-house/tree"
GITHUB_WORKFLOW = "build.yml"
GITHUB_BRANCH = "main"
ARTIFACT_NAME = "firmware"


def usb_run(*args):
    subprocess.run(["mpremote"] + list(args), check=True)


def fetch_github_artifacts(branch: str = GITHUB_BRANCH) -> list[Path]:
    """Fetch latest firmware artifacts from GitHub CI via nightly.link.

    Returns list of paths to downloaded files in a temp directory.
    """
    import httpx

    # nightly.link provides unauthenticated access to GitHub Actions artifacts
    # URL format: https://nightly.link/<owner>/<repo>/workflows/<workflow>/<branch>/<artifact>.zip
    url = f"https://nightly.link/{GITHUB_REPO}/workflows/{GITHUB_WORKFLOW}/{branch}/{ARTIFACT_NAME}.zip"

    print("Fetching latest firmware from GitHub CI...")
    print(f"  URL: {url}")

    try:
        with httpx.Client(follow_redirects=True, timeout=30.0) as client:
            response = client.get(url)
            response.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            print(f"Error: No artifacts found for branch '{branch}'")
            print("Make sure the CI workflow has run successfully on this branch.")
            print(f"Check: https://github.com/{GITHUB_REPO}/actions")
        else:
            print(f"Error fetching artifact: {e}")
        sys.exit(1)
    except httpx.ConnectError as e:
        print(f"Connection error: {e}")
        sys.exit(1)

    # Extract to temp directory
    temp_dir = Path(tempfile.mkdtemp(prefix="tree-firmware-"))
    zip_path = temp_dir / "firmware.zip"
    zip_path.write_bytes(response.content)

    print(f"  Downloaded {len(response.content)} bytes")

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(temp_dir)

    zip_path.unlink()  # Clean up zip file

    # Return all extracted files
    files = list(temp_dir.glob("*"))
    print(f"  Extracted: {', '.join(f.name for f in files)}")

    return files


def ota_push(host: str, port: int, files: list[str], no_reboot: bool = False):
    """Push files to Pico via HTTP OTA."""
    import httpx

    base_url = f"http://{host}:{port}"

    # Check connection
    print(f"Checking {base_url}...")
    try:
        r = httpx.get(f"{base_url}/status", timeout=5.0)
        info = r.json()
        print(f"Connected to {info['hostname']} ({info['ip']})")
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
        print(f"Could not connect to {host}:{port}")
        print("Check: power, WiFi connection, IP address (may have changed)")
        sys.exit(1)

    # Filter to valid files and track if any .py files were pushed
    valid_files = []
    for filepath in files:
        path = Path(filepath)
        if not path.exists():
            print(f"File not found: {filepath}")
            continue
        valid_files.append(path)

    if not valid_files:
        print("No valid files to push")
        sys.exit(1)

    # Push each file (defer reboot until all files are done)
    py_files_pushed = False
    for path in valid_files:
        content = path.read_bytes()
        filename = path.name
        print(f"Pushing {filename} ({len(content)} bytes)...")

        # Don't reboot after each file - we'll reboot at the end
        r = httpx.post(
            f"{base_url}/update?file={filename}&reboot=0",
            content=content,
            headers={"Content-Type": "application/octet-stream"},
            timeout=30.0,
        )

        if r.status_code == 200:
            result = r.json()
            print(f"Success: {result.get('message', 'Updated')}")
            if filename.endswith(".py"):
                py_files_pushed = True
        else:
            print(f"Failed: {r.text}")
            sys.exit(1)

    # Reboot if any .py files were pushed (unless --no-reboot)
    if py_files_pushed and not no_reboot:
        print("Rebooting...")
        try:
            r = httpx.post(f"{base_url}/reboot", timeout=5.0)
            if r.status_code == 200:
                print("Reboot initiated")
            else:
                print(f"Reboot request failed: {r.text}")
        except httpx.ReadTimeout:
            # Expected - device reboots before responding
            print("Reboot initiated")
        except httpx.ConnectError:
            # Also expected if device reboots quickly
            print("Reboot initiated")
    elif no_reboot:
        print("Skipping reboot (--no-reboot)")
    else:
        print("No .py files pushed, skipping reboot")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flash main.py to Pico via USB or OTA (WiFi)."
    )
    parser.add_argument(
        "--run", action="store_true",
        help="Run script on Pico without saving to flash (USB only)"
    )
    parser.add_argument(
        "--ota", action="store_true",
        help="Push via WiFi OTA instead of USB"
    )
    parser.add_argument(
        "--host",
        help="OTA hostname or IP address"
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="OTA port (default: 8080)"
    )
    parser.add_argument(
        "--config", action="store_true",
        help="Push config.py only"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Push all scripts (main.py, ota.py, mqtt.py, config.py)"
    )
    parser.add_argument(
        "--coords", action="store_true",
        help="Include coordinates/coords_compact.txt"
    )
    parser.add_argument(
        "--no-reboot", action="store_true",
        help="Don't reboot after OTA update"
    )
    parser.add_argument(
        "--github", action="store_true",
        help="Fetch latest firmware from GitHub CI artifacts (via nightly.link)"
    )
    parser.add_argument(
        "--branch", default=GITHUB_BRANCH,
        help=f"GitHub branch to fetch from (default: {GITHUB_BRANCH})"
    )
    parser.add_argument(
        "files", nargs="*", default=["main.py"],
        help="Files to flash (default: main.py)"
    )

    args = parser.parse_args()

    # Determine which files to flash
    if args.github:
        # Fetch from GitHub CI artifacts
        downloaded = fetch_github_artifacts(args.branch)
        files = [str(f) for f in downloaded]
        # Note: --all, --config, --coords are ignored when using --github
        # since we use whatever the CI produced
    elif args.all:
        files = ["main.py", "ota.py", "mqtt.py", "config.py"]
        if args.coords:
            files.append("coordinates/coords_compact.txt")
    elif args.config:
        files = ["config.py"]
    else:
        files = args.files
        if args.coords:
            files = files + ["coordinates/coords_compact.txt"]

    if args.ota:
        if not args.host:
            parser.error("--host is required for OTA updates")
        ota_push(args.host, args.port, files, no_reboot=args.no_reboot)
    elif args.run:
        print("Running script on Pico...")
        usb_run("run", "main.py")
    else:
        print(f"Copying {', '.join(Path(f).name for f in files)} to Pico...")
        for f in files:
            dest = Path(f).name  # Use basename for destination
            usb_run("cp", f, f":{dest}")
        print("Done! Script will auto-run on boot.")
