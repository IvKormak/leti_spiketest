from model import *
from utility import *
import numpy as np
import pickle
from os import mkdir

def test(model):
    rotation = [[0, 180, 270, 90], [270, 0, 180, 90]]

    arrows = pickle.load(open('resources\\arrows.bin', 'rb'))

    white_square = np.multiply(np.ones((28,28)), 255)
    current_frame = np.copy(white_square)
    current_arrow = np.copy(white_square)
    frames = []
    frames_shown = 0

    frame = model.next()
    while isinstance(frame, np.ndarray):
        if 1 in frame:
            frames_shown = 0
            current_arrow = arrows[frame[0]]
        if frames_shown > 50:
            current_arrow = np.copy(white_square)
        data = model.raw_data
        synapse = DataFeed.parse_aer(data)[0][0]
        x_coord = int(synapse[0:2], base=16)
        y_coord = int(synapse[2:4], base=16)
        color = (synapse[4] == '0') * 255
        current_frame[y_coord][x_coord] = color
        frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
        frames_shown += 1
        frame = model.next()

    iio.mimwrite('animation2.gif', frames, fps=60)