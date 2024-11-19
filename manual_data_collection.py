import keyboard
import datetime
import uuid
import time
import helper


def take_screenshot():
    folder = "manual_data"
    helper.saveTrainingScreenshot(folder)


def listen(name, pause, end, silent, args):
    if not silent: print(f"{name} started")
    # Start listening for the 'right shift' key press
    keyboard.add_hotkey('right shift', take_screenshot)

    if not silent: print("Listening for 'right shift' key to take a screenshot...")

    while not end[0]:
        time.sleep(0.02)
        pass

    keyboard.remove_hotkey('right shift')
    if not silent: print(f"{name} ended")
