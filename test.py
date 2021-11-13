from model import *
from utility import *
import numpy as np
import imageio as iio
import pickle


f = "C:\\Users\\Кормак\\PycharmProjects\\leti_spiketest\\exp1636276751"
g = "f27.580172105453325.genom"

rotation = [[0, 180, 270, 90], [270, 0, 180, 90]]

arrow_up = iio.imread("arrow.bmp")
arrow_lt = np.rot90(arrow_up)
arrow_dn = np.rot90(arrow_up, 2)
arrow_rt = np.rot90(arrow_up, 3)
arrow_ru = iio.imread("arrow2.bmp")
arrow_lu = np.rot90(arrow_ru)
arrow_ld = np.rot90(arrow_ru, 2)
arrow_rd = np.rot90(arrow_ru, 3)

arrows = [
    arrow_up,
    arrow_dn,
    arrow_lt,
    arrow_rt,
    arrow_lu,
    arrow_ru,
    arrow_ld,
    arrow_rd]

timer = Timer(0)
feed_opts = {'mode': 'file', 'source': 'out.bin'}
feed = CameraFeed(**feed_opts)
model = Network(t=timer, datafeed=feed, structure=(28*28, 8), wta=1, learn=False)
model.set_genom(pickle.load(open(f"{f}/{g}", "rb")))

white_square = np.multiply(np.ones((28,28)), 255)
current_frame = np.copy(white_square)
current_arrow = np.copy(white_square)
frames = []
frames_shown = 0
for frame in model:
    if 1 in frame:
        frames_shown = 0
        current_arrow = arrows[frame.index(1)]
    if frames_shown > 50:
        current_arrow = np.copy(white_square)
    data = model.raw_data
    synapse = parse_aes(data)[0][0]
    x_coord = int(synapse[0:2], base=16)
    y_coord = int(synapse[2:4], base=16)
    color = (synapse[4] == '0') * 255
    current_frame[y_coord][x_coord] = color
    frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
    frames_shown += 1
iio.mimwrite('animation.gif', frames, fps=60)
