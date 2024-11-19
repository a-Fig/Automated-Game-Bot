import time
import keyboard as kb
import random
import pyautogui as gui
import math

import globals as G
import helper
import window_info
import objectFinder as OF
import weapon_spell_casting as cast
import obstacle_instance_detection as OID


def sheild():
    if G.getlastSpellImage().spells[4].class_name == "up":
        kb.press_and_release('d')
        time.sleep(2.5)
        return True
    return False


def attackMob():
    if G.getlastSpellImage().spells[3].class_name == "up":
        kb.press_and_release('r')
        time.sleep(0.5)

    for ob in G.getlastScannedImage().mobs:
        if ob.isReachable():
            helper.clickOnPoint(ob.point)
            kb.press_and_release('space')
            cast.warbow(G.getlastSpellImage().spells)
            return True
        else:
            print(f"{ob.object.class_name} is unreachable")
    return False


def peace():
    for ob in G.getlastScannedImage().peaceItems:
        if (ob.object.class_name == "silver" or ob.object.class_name == "cup of piss") and ob.object.confidence > .55:
            if ob.isReachable():
                print(ob.object.class_name)
                helper.clickOnPoint(ob.point)
                time.sleep(2)
                return True
            else:
                print(f"{ob.object.class_name} is unreachable")
        elif(ob.object.class_name == "wisp caged"):
            if ob.isReachable():
                print(ob.object.class_name)
                helper.clickOnPoint(ob.point)
                for i in range(11):
                    time.sleep(.5)
                    if danger():
                       return True
                return True
            else:
                print(f"{ob.object.class_name} is unreachable")
    return False


def farm():
    for ob in G.getlastScannedImage().farmables:
        if ob.isReachable():
            print("FARM", ob.object.class_name)
            helper.clickOnPoint(ob.point)

            def temp():
                print(".", end='')
                return OF.loading_bar_detected() or danger() or (not window_info.getWindow().isActive)

            print("walking", end='')
            if not helper.wait_unless(3, temp):
                if danger() or (not window_info.getWindow().isActive):
                    return False
            else:
                print("done")

            print("farming", end='')
            while OF.loading_bar_detected():
                print(".", end='')
                if danger():
                   return False
            print("done")
            helper.await_new_image()
            return True
        else:
            print(f"{ob.object.class_name} is unreachable")
    return False


def leash_and_heal():
    if G.getlastSpellImage().spells[5].class_name == "up":
        kb.press_and_release('f')
        time.sleep(3)
        kb.press_and_release('f')
    else:
        return False
    i = 0
    while G.LOW_HEALTH[0] and not danger():
        i += 1
        time.sleep(.5)
    helper.wait_unless(5, danger)
    return True


def eat_food():
    kb.press_and_release('2')
    time.sleep(2)


def danger():
    scanned_image = G.getlastScannedImage()
    if len(scanned_image.reds) > 0:
        print("RED, DANGER")
        return True
    if G.LOW_HEALTH[0]:
        print("LOW, DANGER")
        return True
    for ob in scanned_image.mobs:
        if not helper.is_mob_close(ob):
            pass
            #print("mob is far, no danger")
        elif not ob.isReachable():
            pass
            #print("mob is unreachable")
        else:
            print("MOB, DANGER")
            return True
    return False


def wonder():
    print("wondering")
    xmove = window_info.windowDimensions()[0] * 0.25
    ymove = window_info.windowDimensions()[1] * 0.25

    # this should be improved, its kinda hand wavey, needs to be better defined
    # i like it like this
    def pickRandomPoint(xmove, ymove):
        rand_x = random.randint(0, int(xmove)) * (math.pow(-1, random.randint(1, 2)))
        rand_y = random.randint(0, int(ymove)) * (math.pow(-1, random.randint(1, 2)))
        mainplayer = G.getlastScannedImage().ref_point

        windowLocation = window_info.windowLocation()
        windowDimensions = window_info.windowDimensions()
        return OID.Point((mainplayer[0] + rand_x) % (windowDimensions[0] + windowLocation[0]), (mainplayer[1] + rand_y) % (windowDimensions[1] + windowLocation[1]))

    random_point = pickRandomPoint(xmove, ymove)
    while not helper.isObjectReachable(random_point) and not helper.are_points_within_range(random_point, helper.tupleToPoint(G.getlastScannedImage().ref_point), .15):
        print(f"{random_point.x},{random_point.y} is not reachable")
        random_point = pickRandomPoint(xmove, ymove)

    print("wonder point chosen")
    gui.moveTo(random_point.x, random_point.y, duration=0.2)
    for i in range(0, 4):
        gui.rightClick()
        time.sleep(1.5)
        if helper.activity() or not helper.isObjectReachable(random_point) or not window_info.getWindow().isActive:
            break
