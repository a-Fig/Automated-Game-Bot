import math
import random
import uuid
import pyautogui as gui
import keyboard as kb
import time
import globals as G
import obstacle_instance_detection as OID
import window_info


def is_mob_close(mob, prange=.3):
    return are_points_within_range(mob.point, tupleToPoint(G.getlastScannedImage().ref_point), prange)


# check if a given point is withn x% of the window height of the window center
def are_points_within_range(given_point, center_point, prange):
    def distance_between_points(point1, point2):
        return math.sqrt((point2.x - point1.x) ** 2 + (point2.y - point1.y) ** 2)
    # are points within p% of window height
    # height% > distance from given - center
    #print(f"{window_info.windowDimensions()[1] * prange:.2f} > {(distance_between_points(given_point, center_point)):.2f}")
    return (window_info.windowDimensions()[1] * prange) > (distance_between_points(given_point, center_point))


def await_new_image():
    now = time.time()
    while now > G.getlastScannedImage().timestamp:
        time.sleep(.05)


def find_corner(redObject):
    xmultiplier = math.pow(-1, random.randint(1, 2))
    ymultiplier = math.pow(-1, random.randint(1, 2))

    return redObject.x + (xmultiplier * redObject.width/2), redObject.y + (ymultiplier * redObject.height/2)


unique_id = str(uuid.uuid4())
default_save_folder = "bridgewatch_data"
screenshot_counter = 1


def attemptScreenShot(odds=200, folder=default_save_folder, image_name=""):
    if random.randint(1, odds) == 1:
        saveTrainingScreenshot(folder, image_name)
        return True
    return False


def saveTrainingScreenshot(folder=default_save_folder, image_name="", whole_screen=False):
    global screenshot_counter

    image_name = f"{image_name}_{str(uuid.uuid4())}"
    filename = f"training_data/{folder}/{image_name}.png"

    screenshot = gui.screenshot() if whole_screen else window_info.albion_screenshot()
    screenshot.save(filename)

    screenshot_counter += 1

    print(f"Screenshot saved to {folder} as {image_name}")


def tupleToPoint(_tuple):
    return OID.Point(_tuple[0], _tuple[1])


def activity(scanned_image=0):
    if scanned_image == 0:
        scanned_image = G.getlastScannedImage()

    if len(scanned_image.reds) > 0:
        return True
    for ob in scanned_image.mobs:
        if ob.isReachable():
            return True
    for ob in scanned_image.farmables:
        if ob.isReachable():
            return True
    for ob in scanned_image.peaceItems:
        if ob.isReachable():
            return True
    return False


def clickOnPoint(point):
    clickOn(point.x, point.y)


def clickOn(x, y):
    try:
        kb.press_and_release('s')
        gui.moveTo(x, y, duration=0.2)
        gui.click()
    except gui.FailSafeException:
        print("Fail-safe triggered! Mouse moved to a corner of the screen.")
        pause()

END_PROGRAM = False
def pause():
    global END_PROGRAM
    G.GENERAL_SCAN_PROCESSES.switch()
    G.MULTIUSE_SCAN_PROCESSES.switch()

    print("pausing program")
    print("type 'stop' or 'end' to end excution")
    print('>',end='')
    userinput = input()

    if userinput == "stop" or userinput == "end":
        print("ending program...")
        END_PROGRAM = True
    else:
        G.GENERAL_SCAN_PROCESSES.switch()
        G.MULTIUSE_SCAN_PROCESSES.switch()
        print("starting program in...")
        for i in range(1, 5):
            print(5 - i)
            time.sleep(1)
        END_PROGRAM = False
    return not END_PROGRAM


def attemptArmorCast(spells):
    armor_spell_cast = False
    if spells[3].class_name == "up":
        kb.press_and_release('r')
        time.sleep(0.5)
        armor_spell_cast = True
    if spells[5].class_name == "up":
        kb.press_and_release('f')
        time.sleep(0.5)
        armor_spell_cast = True
    return armor_spell_cast


def isObjectReachable(point):
    obstacle_image = G.getlastObstacleImage()
    return not (obstacle_image.does_intersect(OID.Line(point, tupleToPoint(G.getlastScannedImage().ref_point))))


def pre_pause_check():
    d = 3
    kb.press_and_release('esc')
    time.sleep(.5)
    for n in range(0,d):
        if not len(G.getlastScannedImage().RawDetectedObjects) > 0:
            attemptScreenShot(5,"nothing_detected")
            print(f"{n+1}/{d} 0 objects detected, preparing to pause execution")
            time.sleep(1)
        else:
            saveTrainingScreenshot()
            return False
    saveTrainingScreenshot(folder="nothing_detected")
    return True


def rate_stats():
    if len(G.GENERAL_IMAGESTACK) > 10:
        print(f"{G.GENERAL_SCAN_PROCESSES.run_speed():.2f}s for last gen run")
        print(f"{10 / (time.time() - G.GENERAL_IMAGESTACK[-10].timestamp):.2f} gen/second")
    if len(G.SPELL_IMAGESTACK) > 10:
        print(f"{G.MULTIUSE_SCAN_PROCESSES.run_speed():.2f}s for last multi run")
        print(f"{10 / (time.time() - G.SPELL_IMAGESTACK[-10].timestamp):.2f} spell/second")


def await_threads():
    print("waiting for scan threads to begin")
    while(not len(G.GENERAL_IMAGESTACK) > 0):
        print("waiting for general thread to begin")
        time.sleep(2)
    while (not len(G.SPELL_IMAGESTACK) > 0):
        print("waiting for spell thread to begin")
        time.sleep(2)
    print("all threads have begun")


class stopwatch():
    def __init__(self):
        self.start_time = time.time()

    def time_passed(self):
        #returns minutes since start
        return (time.time() - self.start_time) / 60

    def restart(self):
        self.start_time = time.time()


def wait_unless(wait_seconds, trigger):
    end_time = time.time() + wait_seconds
    while time.time() < end_time:
        if trigger():
            print("triggered")
            return False
        time.sleep(0.02)
    #print("waited full time")
    return True


