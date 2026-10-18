# Recovery layer: installed over USB only, never written by OTA.
#
# Counts the boot, picks the firmware slot (rolling back or entering safe mode
# after failed boots) and runs it. See bootmgr.launch().
import bootmgr

bootmgr.launch()
