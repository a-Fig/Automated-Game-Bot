import time
import helper
import globals as G
import window_info
import objectFinder as OF
import player_actions as ACT

if __name__ == '__main__':
    print("main")
    OF.ort_info()
    G.MANUAL_DATA.start()
    G.MULTIUSE_SCAN_PROCESSES.start()
    G.GENERAL_SCAN_PROCESSES.start()
    albion_window = window_info.awaitAlbion()

    helper.await_threads()
    bored_counter = 0

    image_rate_testing = False

    time_to_eat = helper.stopwatch()

    run = True
    iterations = 0
    while not image_rate_testing and run and not helper.END_PROGRAM:
        while not albion_window.isActive and run:
            print(f"{window_info.window_title} is not in focus")
            run = helper.pause()

        helper.attemptScreenShot(200, image_name="main")
        print()
        print("LOW" if G.LOW_HEALTH[0] else "gucci")

        OF.printObjects(G.getlastScannedImage().RawDetectedObjects)
        skip = not run
        if (not skip) and not len(G.getlastScannedImage().RawDetectedObjects) > 0:
            skip = True
            if helper.pre_pause_check():
                run = helper.pause()
        if (not skip) and len(G.getlastScannedImage().reds) > 0:
            print("sheild()")
            skip = ACT.sheild()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and G.LOW_HEALTH[0]:
            print("leash_and_heal()")
            skip = ACT.leash_and_heal()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and len(G.getlastScannedImage().mobs) > 0 and helper.is_mob_close(G.getlastScannedImage().mobs[0]):
            print("attackMob() CLOSE")
            skip = ACT.attackMob()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and len(G.getlastScannedImage().peaceItems) > 0:
            print("peace()")
            skip = ACT.peace()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and len(G.getlastScannedImage().farmables) > 0:
            print("farm()")
            skip = ACT.farm()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and len(G.getlastScannedImage().mobs) > 0:
            print("attackMob() FAR")
            skip = ACT.attackMob()
            print("success" if skip else "fail")
            bored_counter = 0 if skip else bored_counter
        if (not skip) and time_to_eat.time_passed() > 30:
            time_to_eat.restart()
            ACT.eat_food()
        else:
            print(f"ate {time_to_eat.time_passed():.0f} mins ago")
        if (not skip):
            print("BORED", bored_counter)
            bored_counter += 1
            helper.attemptScreenShot(20, image_name="bored")
            if bored_counter > 10:
                print("wonder()")
                ACT.wonder()
            time.sleep(.15)

        helper.rate_stats()

        if iterations % 300 == 200:
            print("resetting stacks")
            temp = G.getlastScannedImage()
            with G.STACK_LOCK:
                del G.GENERAL_IMAGESTACK[:]
                G.GENERAL_IMAGESTACK.append(temp)

            temp = G.getlastSpellImage()
            with G.SPELL_LOCK:
                del G.SPELL_IMAGESTACK[:]
                G.SPELL_IMAGESTACK.append(temp)
            print("stacks reset")

        iterations += 1

    while image_rate_testing:
        time.sleep(5)
        print()
        helper.rate_stats()

    print("joining threads")
    G.MULTIUSE_SCAN_PROCESSES.join()
    G.GENERAL_SCAN_PROCESSES.join()
    G.MANUAL_DATA.join()
    print("goodnight")
