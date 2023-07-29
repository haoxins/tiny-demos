import numpy as np
import torch

import time


class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.time = time.time()

    def stop(self):
        self.times.append(time.time() - self.time)
        return self.times[-1]

    def avg(self):
        return self.sum() / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.cumsum(self.times).tolist()


def bench():
    n = 10_000_00
    a = torch.ones(n)
    b = torch.ones(n)
    c = torch.zeros(n)

    timer = Timer()

    for i in range(n):
        c[i] = a[i] + b[i]

    timer.stop()

    print(f"{timer.sum():.5f}")

    timer.start()
    d = a + b
    print(f"{timer.stop():.5f}")


bench()
