"""Tree firmware application. Lives in a firmware slot and is replaced by
updates; the recovery layer (boot.py, main.py, bootmgr.py, safemode.py) is not."""

# Oldest recovery-layer API (bootmgr.API) this app runs on; the build puts it
# in the manifest, and an older recovery layer refuses the bundle. Raise it
# only when the app can't fall back (see recovery.py).
RECOVERY_API = 2
