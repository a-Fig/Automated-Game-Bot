import keyboard as kb
import time

def longbow(spells):
    if spells[1].class_name == "up":
        kb.press_and_release('w')
        time.sleep(.5)
    elif (spells[2].class_name == "up"):
        kb.press_and_release('e')
        time.sleep(1.5)
    else:
        kb.press_and_release('q')
        time.sleep(.5)

def mistpiercer(spells):
    if (spells[2].class_name == "up"):
        kb.press_and_release('e')
        time.sleep(0.61)
    elif spells[1].class_name == "up":
        kb.press_and_release('w')
        time.sleep(.5)
    else:
        kb.press_and_release('q')
        time.sleep(.5)

def warbow(spells):
    if (spells[2].class_name == "up"):
        kb.press_and_release('e')
        time.sleep(0.5)
    elif spells[1].class_name == "up":
        kb.press_and_release('w')
        time.sleep(.5)
    else:
        kb.press_and_release('q')
        time.sleep(.5)


def nature(spells):
    if (spells[2].class_name == "up"):
        kb.press_and_release('e')
        time.sleep(0.1)
    elif (spells[1].class_name == "up"):
        kb.press_and_release('w')
        time.sleep(0.3)
    elif (spells[0].class_name == "up"):
        for j in range(3):
            kb.press_and_release('q')
            time.sleep(0.5)
    else:
        for j in range(4):
            kb.press_and_release('q')
            time.sleep(0.5)
