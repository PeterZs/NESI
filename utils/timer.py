import torch
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def colorize_time(elapsed):
    if elapsed > 1e-3:
        return bcolors.FAIL + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-4:
        return bcolors.WARNING + "{:.3e}".format(elapsed) + bcolors.ENDC
    elif elapsed > 1e-5:
        return bcolors.OKBLUE + "{:.3e}".format(elapsed) + bcolors.ENDC
    else:
        return "{:.3e}".format(elapsed)


class PerfTimer():
    def __init__(self, activate=False):
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()
        self.counter = 0
        self.activate = activate

    def reset(self):
        self.counter = 0
        self.prev_time = time.process_time()
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.prev_time_gpu = self.start.record()

    def check(self, name=None):
        if self.activate:
            cpu_time = time.process_time() - self.prev_time
            cpu_time = colorize_time(cpu_time)

            self.end.record()
            torch.cuda.synchronize()

            gpu_time = self.start.elapsed_time(self.end) / 1e3
            gpu_time = colorize_time(gpu_time)
            if name:
                print("CPU Checkpoint {}: {} s".format(name, cpu_time))
                print("GPU Checkpoint {}: {} s".format(name, gpu_time))
            else:
                print("CPU Checkpoint {}: {} s".format(self.counter, cpu_time))
                print("GPU Checkpoint {}: {} s".format(self.counter, gpu_time))

            self.prev_time = time.process_time()
            self.prev_time_gpu = self.start.record()
            self.counter += 1
            return cpu_time, gpu_time