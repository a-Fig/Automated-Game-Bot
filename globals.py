import multiprocessing_class as MPC
import objectFinder as OF
import threading as th
import manual_data_collection as MDC

GENERAL_IMAGESTACK = list()
STACK_LOCK = th.Lock()
SPELL_IMAGESTACK = list()
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


def pause_switch():
    GENERAL_SCAN_PROCESSES.switch()
    MULTIUSE_SCAN_PROCESSES.switch()


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