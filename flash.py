#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpremote"]
# ///
"""Flash main.py to Pico and optionally run it."""

import subprocess
import sys

def run(*args):
    subprocess.run(["mpremote"] + list(args), check=True)

if __name__ == "__main__":
    if "--run" in sys.argv:
        print("Running script on Pico...")
        run("run", "main.py")
    else:
        print("Copying main.py to Pico...")
        run("cp", "main.py", ":main.py")
        print("Done! Script will auto-run on boot.")
        print("Use --run to execute immediately without flashing.")
