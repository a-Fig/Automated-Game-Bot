import random
import threading as th
import time


class ThreadHandler:
    def __init__(self, name, count, _target, _args, silent=True):
        self.processes = list()
        self.name = name
        self.end = [False]
        self.pause_thread = [False]
        for i in range(0, count):
            self.processes.append(th.Thread(target=_target, args=(name + '#' + str(i+1), self.pause_thread, self.end, silent, _args)))

    def start(self):
        count = 0
        for process in self.processes:
            process.start()
            count += 1

    def pause(self):
        self.pause_thread[0] = True

    def unpause(self):
        self.pause_thread[0] = False

    def switch(self):
        self.pause_thread[0] = not self.pause_thread[0]

    def join(self):
        self.end[0] = True
        self.pause_thread[0] = False
        count = 0
        for process in self.processes:
            process.join()
            count += 1


def repeated_execution(target, name, pause, end, silent, target_run_time, *args):
    #print(target)
    #print(*args)
    #print(name)

    if not silent: print(f"{name} started")
    while not end[0]:
        if pause[0]:
            if not silent: print(f"{name} paused")
            while pause[0]:
                time.sleep(1)
            if not silent: print(f"{name} un-paused")
        before = time.time()
        target(*args)
        target_run_time[0] = time.time() - before
        #if random.randint(0, 9) == 0: print(f"{time.time() - before:.2f}s {name}")
        #time.sleep(3)
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


'''
class ProcessHandler:
    def __init__(self, name, count, target, args):
        self.processes = list()
        self.name = name
        self.end = [False]
        self.pause_thread = [False]
        args = args + (self.pause_thread, self.end)
        for i in range(0,count):
            self.processes.append(mp.Process(target=target, args=args))

    def start(self):
        count = 0
        for process in self.processes:
            process.start()
            count += 1
            print(f"{self.name}#{count} process started")

    def pause(self):
        self.pause_thread[0] = True

    def unpause(self):
        self.pause_thread[0] = False

    def switch(self):
        self.pause_thread = not self.pause_thread

    def join(self):
        count = 0
        for process in self.processes:
            process.join()
            count += 1
            print(f"{self.name}#{count} process joined")
'''
