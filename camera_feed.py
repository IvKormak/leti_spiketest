from utility import *

from math import sin, pi
import numpy as np
import imageio as iio
from random import random


class CameraFeed(object):
    """docstring for CameraFeed"""

    def __init__(self, source=(), x_dim=28, y_dim=28, mode='tuple', frames=-1, start = 0):
        super(CameraFeed, self).__init__()
        if mode == 'file':
            self.file = open(source, 'r')
            self.length = 10
            self.datastream = self._file_read_
            self.pixels = []
            for k in range(x_dim):
                for m in range(y_dim):
                    for p in range(2):
                        pixel = format(k, '02x')+format(m, '02x')+str(p*8)
                        self.pixels.append(pixel)
            self.frames = frames
            self.frames_max = frames
            self.start = 0
            if start:
                self.start = start
                for i in range(start):
                    self.read()

    def reset(self):
        self.file.seek(0, 0)
        self.frames = self.frames_max
        if self.start:
            for i in range(self.start):
                self.read()

    def read(self):
        return self.datastream()

    def get_pixels(self):
        return list(self.pixels)

    def _file_read_(self):
        if self.frames > 0:
            self.frames -= 1
        string = ""
        hexadecimals = "0123456789abcdef"
        while char := self.file.read(1):
            if char in hexadecimals:
                string += char
            if len(string) == self.length:
                break
        if string == '' or not(self.frames):
            raise StopIteration
        return string


if __name__ == "__main__":
    feed = CameraFeed(mode='file', source="out.bin")
    feed.reset()
    matrix = np.zeros((28,28))
    array = []
    num = 0
    while 1:
        try:
            spike = feed.read()
            synapse = parse_aes(spike)[0][0]
            x_coord = int(synapse[0:2], base=16)
            y_coord = int(synapse[2:4], base=16)
            color = (synapse[4]!='0')*255
            matrix[y_coord][x_coord] = color
            num = num+1
            #iio.imwrite('raw/'+str(num)+'.png', matrix)
            array.append(matrix.copy())
        except:
            break
    iio.mimwrite('animation.gif', array, duration=0.01)

