import multiprocessing_class as MPC
import objectFinder as OF
import threading as th
from collections import deque
import manual_data_collection as MDC

# bounded, so old scans drop off on their own instead of being reset by hand
GENERAL_IMAGESTACK = deque(maxlen=50)
STACK_LOCK = th.Lock()
SPELL_IMAGESTACK = deque(maxlen=50)
SPELL_LOCK = th.Lock()
LOW_HEALTH = [False]

GENERAL_SCAN_PROCESSES = MPC.RepeaterThread(
    "general_scan", 2,
    OF.general_scanner,
    (GENERAL_IMAGESTACK, STACK_LOCK),
    False
)
MULTIUSE_SCAN_PROCESSES = MPC.RepeaterThread(
    "multiuse_scan", 1,
    OF.multiuse_scanner,
    (SPELL_IMAGESTACK, SPELL_LOCK, LOW_HEALTH),
    False
)
MANUAL_DATA = MPC.ThreadHandler("manual_data", 1, MDC.listen, (), False)


# explicit pause and resume, not a toggle, so the scanners can never end up
# in the opposite state from the one the main loop thinks they are in
def pause_scanners():
    GENERAL_SCAN_PROCESSES.pause()
    MULTIUSE_SCAN_PROCESSES.pause()


def resume_scanners():
    GENERAL_SCAN_PROCESSES.unpause()
    MULTIUSE_SCAN_PROCESSES.unpause()


def getlastScannedImage():
    with STACK_LOCK:
        last_scanned_image = GENERAL_IMAGESTACK[-1]
    return last_scanned_image


def getlastSpellImage():
    with SPELL_LOCK:
        last_scanned_image = SPELL_IMAGESTACK[-1]
    return last_scanned_image


def getlastObstacleImage():
    return getlastScannedImage().detectedObstacles