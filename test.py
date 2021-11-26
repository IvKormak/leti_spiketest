from model import *
from utility import *
import numpy as np
import imageio as iio
import pickle
from os import mkdir


f = "C:\\Users\\Кормак\\PycharmProjects\\leti_spiketest\\exp1637728982"
g = "f.weight"

rotation = [[0, 180, 270, 90], [270, 0, 180, 90]]

arrows = pickle.load(open('resources\\arrows.bin', 'rb'))

model, timer = init_model(learn = False, structure=[7,])

folder = f"exp{int(time.time())}"
mkdir(folder)

model.set_weights(pickle.load(open(f"{f}/{g}", "rb")))

white_square = np.multiply(np.ones((28,28)), 255)
current_frame = np.copy(white_square)
current_arrow = np.copy(white_square)
frames = []
frames_shown = 0

while frame := model.next():
    if 1 in frame:
        frames_shown = 0
        current_arrow = arrows[frame.index(1)]
    if frames_shown > 50:
        current_arrow = np.copy(white_square)
    data = model.raw_data
    synapse = parse_aer(data)[0][0]
    x_coord = int(synapse[0:2], base=16)
    y_coord = int(synapse[2:4], base=16)
    color = (synapse[4] == '0') * 255
    current_frame[y_coord][x_coord] = color
    frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
    frames_shown += 1

iio.mimwrite('animation2.gif', frames, fps=60)