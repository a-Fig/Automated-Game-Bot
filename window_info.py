import pygetwindow as gw
import time
import random
import pyautogui as gui

window_title = "Albion Online Client"
ALBION_WINDOW = -1


def getWindow():
    global ALBION_WINDOW
    if ALBION_WINDOW == -1:
        windows = gw.getWindowsWithTitle(window_title)
        if (len(windows) > 0):
            ALBION_WINDOW = windows[0]
        else:
            ALBION_WINDOW = awaitAlbion()
    return ALBION_WINDOW


def awaitAlbion():
    while not len(gw.getWindowsWithTitle(window_title)) > 0:
        print(f"waiting for {window_title} to open")
        time.sleep(2)

    print(f"{window_title} found")
    return gw.getWindowsWithTitle(window_title)[0]


def randomPointOnWindow():
    _windowDimensions = windowDimensions()
    _windowLocation = windowLocation()
    rx = random.randint(0, _windowDimensions[0]) + _windowLocation[0]
    ry = random.randint(0, _windowDimensions[1]) + _windowLocation[1]
    return rx, ry


def windowCenter():
    window = getWindow()
    position = window.topleft
    return (position[0] + (window.width//2), position[1] + (window.height//2))


def windowDimensions():
    window = getWindow()
    return (window.width, window.height)


def windowLocation():
    return getWindow().topleft


def albion_screenshot():
    window = getWindow()
    region_screenshot = gui.screenshot(region=(window.topleft[0], window.topleft[1], window.width, window.height))
    return region_screenshot