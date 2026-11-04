import threading as th
import time
import traceback


class ThreadHandler:
    def __init__(self, name, count, _target, _args, silent=True):
        self.processes = list()
        self.name = name
        self.end = [False]
        self.pause_thread = [False]
        for i in range(0, count):
            self.processes.append(th.Thread(target=_target, args=(name + '#' + str(i+1), self.pause_thread, self.end, silent, _args)))

    def start(self):
        for process in self.processes:
            process.start()

    def pause(self):
        self.pause_thread[0] = True

    def unpause(self):
        self.pause_thread[0] = False

    def switch(self):
        self.pause_thread[0] = not self.pause_thread[0]

    def join(self):
        self.end[0] = True
        self.pause_thread[0] = False
        for process in self.processes:
            process.join()


def repeated_execution(target, name, pause, end, silent, target_run_time, *args):
    if not silent: print(f"{name} started")
    while not end[0]:
        if pause[0]:
            if not silent: print(f"{name} paused")
            while pause[0]:
                time.sleep(1)
            if not silent: print(f"{name} un-paused")
        before = time.time()
        try:
            target(*args)
        except Exception:
            # without this, one bad frame kills the scanner thread and the main loop
            # keeps acting on the last image it produced, forever
            print(f"{name} hit an error, retrying:")
            traceback.print_exc()
            time.sleep(0.5)
            continue
        target_run_time[0] = time.time() - before
    if not silent: print(f"{name} ended")


class RepeaterThread(ThreadHandler):
    def __init__(self, name, count, _target, _args, silent=True):
        self.processes = list()
        self.name = name
        self.end = [False]
        self.pause_thread = [False]
        self.target_run_time = [0]
        for i in range(0, count):
            self.processes.append(th.Thread(target=repeated_execution, args=(_target, name + '#' + str(i+1), self.pause_thread, self.end, silent, self.target_run_time, *_args)))

    def run_speed(self):
        return self.target_run_time[0]
