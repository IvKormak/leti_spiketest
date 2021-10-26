from math import sin, pi
from random import random
from utility import *


class CameraFeed(object):
    """docstring for CameraFeed"""

    def __init__(self, datastream=(), timescale=100, frames=1000, period=100, mode='tuple'):
        super(CameraFeed, self).__init__()
        if mode == 'bnw':
            x_dim, y_dim = 1, 8
            self.pixels = {
                polarity + ((y + (x << 8)) << 1)
                for x in range(1, x_dim + 1)
                for y in range(1, y_dim + 1)
                for polarity in range(2)
            }
            self.datastream = self._blackandwhite_(frames, period, timescale, x_dim, y_dim)
        elif mode == 'tuple':
            self.datastream = iter(datastream)

    def read(self):
        return next(self.datastream)

    def get_pixels(self):
        return list(self.pixels)

    def _blackandwhite_(self, frames, period, timescale, x_dim, y_dim):
        for frame in range(1, frames + 1):
            for y in range(1, y_dim + 1):
                for x in range(1, x_dim + 1):
                    n = (frame - 1) * (x_dim) * (y_dim) + (y - 1) * (x_dim) + x - 1
                    phase = sin(frame / period * 2 * pi) / 2
                    polarity = round(random() / 2 + phase)
                    address = polarity + ((y + (x << 8)) << 1)
                    time = n * timescale & 0x3fffff
                    combined = (address << 22) + time
                    yield (combined)


if __name__ == "__main__":
    feed = CameraFeed(mode='bnw', frames=1000, timescale=30)
    for spike in feed.datastream:
        print(format(spike >> 22, '#017b'), spike & 0x3fffff)
        print(format(spike, '#040b'))