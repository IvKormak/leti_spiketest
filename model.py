from layer import Layer
from utility import *
from camera_feed import CameraFeed

import matplotlib.pyplot as plt
import numpy as np


class Network(object):
    """docstring for Network"""

    def __init__(self, timer: Timer, datafeed: CameraFeed, synapses: tuple, structure: tuple, **kwargs):
        super(Network, self).__init__()

        self.timer = timer
        timer.add_listener(self)

        self.layers = []
        self.frame = 0
        self.clock = -1
        self.feed = datafeed
        self.structure = structure

        for layer_num in range(len(self.structure)):
            genom = {x: 100 for x in synapses}
            layer = Layer(
                    timer=timer,
                    neuron_number=self.structure[layer_num],
                    layer_number=layer_num + 1,
                    genom=genom,
                    **kwargs
            )
            synapses = layer.get_synapses()
            self.layers.append(layer)

    def update_clock(self, time):
        self.clock = time

    def __next__(self):
        raw_data = self.feed.read()
        input_data, time = parse_aes(raw_data)  # cutting bits 0:22
        self.timer.set_time(time)  # устанавливаем часы по битам 22:0 входного пакета данных
        for layer in self.layers:
            layer.tick(input_data)
            input_data = layer.get_fired_synapses()
        self.frame += 1
        return input_data

    def __iter__(self):
        return self

    def get_journals_from_layer(self, layer=-1):
        return self.layers[layer].get_journals()

    def get_genoms_from_layer(self, layer=-1):
        return self.layers[layer].get_genoms()


def plot_output():
    xdata, ydata = [], []
    for data in model.get_journals_from_layer():
        xdata = [k for k in data]
        ydata.append([data[k] for k in data])
    #        ax.plot(xdata, ydata, color='C'+str(i))
    ydata = [[ydata[j][i] for j in range(len(ydata))] for i in range(len(ydata[0]))]
    l = plt.plot(xdata, ydata)
    plt.draw()


if __name__ == "__main__":

    feed = CameraFeed(mode='bnw', frames=1000, timescale=30, period=240)
    t = Timer(0)
    model = Network(timer=t, datafeed=feed, synapses=feed.get_pixels(), structure=(28, 8), i_thres=1000)

    fig, ax = plt.subplots()
    for output in model:
        pass
    plot_output()
    plt.show()
