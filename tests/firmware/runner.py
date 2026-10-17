# One simulated boot: what the rp2 port does at reset, minus the hardware.
# Run by devsim.py with the device directory as the working directory.
import os
import sys

# Like the Pico: the flash root (cwd) first, then the mocks for machine etc.
sys.path[0] = ""
sys.path.insert(1, os.getenv("DEVSIM_MOCKS"))

g = {"__name__": "__main__"}
for name in ("boot.py", "main.py"):
    try:
        src = open(name).read()
    except OSError:
        continue
    exec(compile(src, name, "exec"), g)
