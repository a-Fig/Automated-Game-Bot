import os
import inference
import pyautogui
import math
import time
import obstacle_instance_detection as OID
import window_info
import onnxruntime as ort
from spell_slots import SpellSlotMapper

# set these as environment variables so the keys stay out of the source
ROBOFLOW_KEY1 = os.environ.get("ROBOFLOW_KEY1", "xxxxxxxxxxxxx")
ROBOFLOW_KEY2 = os.environ.get("ROBOFLOW_KEY2", "xxxxxxxxxxxxx")

GENERAL_ALBION_MODEL = inference.get_model("albion_detector-l4o1a/5", ROBOFLOW_KEY2)
ALBION_SPELL_MODEL = inference.get_model("spell_detector/4", ROBOFLOW_KEY1)
ALBION_OBSTACLE_MODEL = inference.get_model("obstacle_instance_detection/5", ROBOFLOW_KEY2)
ALBION_LOADING_MODEL = inference.get_model("loading_bar/1", ROBOFLOW_KEY2)
ALBION_HEALTH_MODEL = inference.get_model("low_detector/1", ROBOFLOW_KEY2)

def ort_info():
    print(ort.cuda_version)
    print(ort.get_available_providers())
    print(ort.get_device())


def getModel():
    return GENERAL_ALBION_MODEL


def mainPlayerLocation(objects_con):
    for ob in objects_con:
        if ob.class_name == "player":
            offset = window_info.windowLocation()
            return ob.x + offset[0], ob.y + offset[1]

    center = window_info.windowCenter()
    return (center[0] + 2, center[1] - 95)


def sort_by_distance(points, rx, ry):
    def distance(point):
        return math.sqrt((point.x - rx) ** 2 + (point.y - ry) ** 2)

    return sorted(points, key=distance)


class DetectedObject:
    def __init__(self, object, detectedObstacles, ref_point):
        self.object = object
        self.point = OID.Point(object.x + window_info.windowLocation()[0], object.y + window_info.windowLocation()[1])
        self.detectedObstacles = detectedObstacles
        self.line = OID.Line(OID.Point(ref_point[0], ref_point[1]),self.point)
        self.reachable = -1 #-1 is unset, 0 is False, 1 is True

    def isReachable(self):
        if self.reachable == -1:
            self.reachable = 0 if self.detectedObstacles.does_intersect(self.line) else 1
        return self.reachable == 1


class AutoParsedScannedImage:
    def __init__(self):
        image = window_info.albion_screenshot()
        self.timestamp = time.time()

        self.RawDetectedObjects = scanImage(image)

        self.ref_point = mainPlayerLocation(self.RawDetectedObjects)
        self.RawDetectedObjects = sort_by_distance(self.RawDetectedObjects, self.ref_point[0], self.ref_point[1])

        self.detectedObstacles = OID.ManualObstacleImage(image, self.timestamp)

        self.players = list()
        self.mobs = list()
        self.reds = list()
        self.farmables = list()
        self.peaceItems = list()
        for ob in self.RawDetectedObjects:
            if(ob.confidence > .65):
                if(ob.class_name == "player"):
                    self.players.append(DetectedObject(ob, self.detectedObstacles,self.ref_point))
                elif(ob.class_name == "mob"):
                    self.mobs.append(DetectedObject(ob, self.detectedObstacles,self.ref_point))
                elif (ob.class_name == "red"):
                    self.reds.append(DetectedObject(ob, self.detectedObstacles,self.ref_point))
                elif ob.class_name == "hide" or ob.class_name == "fiber":
                    self.farmables.append(DetectedObject(ob, self.detectedObstacles,self.ref_point))
                elif(ob.class_name == "silver" or ob.class_name == "cup of piss" or ob.class_name == "wisp caged"):
                    self.peaceItems.append(DetectedObject(ob, self.detectedObstacles,self.ref_point))
        self.elapsed_time = time.time() - self.timestamp


def general_scanner(image_stack, stack_lock):
    new_scanned_image = AutoParsedScannedImage()
    with stack_lock:
        # two scanner threads run at once, so a slow scan can finish after a newer one.
        # only append if it is newer, so image_stack[-1] is always the newest screenshot
        if len(image_stack) == 0 or new_scanned_image.timestamp >= image_stack[-1].timestamp:
            image_stack.append(new_scanned_image)


def multiuse_scanner(image_stack, stack_lock, low_health):
    # spell stuff
    new_spell_image = SpellImage()
    with stack_lock:
        image_stack.append(new_spell_image)

    time.sleep(.2)

    # health stuff
    low_health[0] = ALBION_HEALTH_MODEL.infer(window_info.albion_screenshot())[0].predictions[0].class_name == "low"

    time.sleep(.2)

    new_spell_image = SpellImage()
    with stack_lock:
        image_stack.append(new_spell_image)

    time.sleep(.6)


class SpellImage:
    def __init__(self):
        self.timestamp = time.time()
        self.image = 0
        self.spells = create_spell_image(window_info.albion_screenshot())
        self.elapsed_time = time.time() - self.timestamp

def spellScan():
    return create_spell_image(window_info.albion_screenshot())


SPELL_SLOTS = SpellSlotMapper()


def create_spell_image(image):
    return SPELL_SLOTS.map(ALBION_SPELL_MODEL.infer(image)[0].predictions)


def loading_bar_detected():
    return len(ALBION_LOADING_MODEL.infer(window_info.albion_screenshot())[0].predictions) > 0


def screenshot():
    return pyautogui.screenshot()


def scanImage(img):
    response = getModel().infer(image=img)
    return response[0].predictions


def scanScreen():
    response = getModel().infer(window_info.albion_screenshot())
    return response[0].predictions


def printObjects(objects, min_confidence = .4):
    print('[',end='')
    for object in objects:
        if object.confidence >= min_confidence:
            print(f"{object.class_name}[c:{object.confidence * 100:.2f} x:{object.x} y:{object.y}] ",end='')
    print(']')


def scan():
     while True:
        img = window_info.albion_screenshot()
        objects = scanImage(img)
        if len(objects) > 0:
            printObjects(objects)


