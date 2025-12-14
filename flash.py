#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = ["mpremote"]
# ///
"""Flash pico_poc.py to Pico as main.py and optionally run it."""

import subprocess
import sys

def run(*args):
    subprocess.run(["mpremote"] + list(args), check=True)

if __name__ == "__main__":
    if "--run" in sys.argv:
        print("Running script on Pico...")
        run("run", "pico_poc.py")
    else:
        print("Copying pico_poc.py -> main.py on Pico...")
        run("cp", "pico_poc.py", ":main.py")
        print("Done! Script will auto-run on boot.")
        print("Use --run to execute immediately without flashing.")
