from layer import Layer
from utility import *
from camera_feed import CameraFeed
from timer import *

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
from random import shuffle
from os import mkdir


class Network(object):
    """docstring for Network"""

    def __init__(self, t: Timer, datafeed: CameraFeed, structure: tuple, **kwargs):
        super(Network, self).__init__()

        self.timer = t
        t.add_listener(self)

        self.layers = []
        self.frame = 0
        self.clock = -1
        self.feed = datafeed
        self.structure = structure

        self.kwargs = kwargs

        synapses = self.feed.get_pixels()

        for layer_num in range(len(self.structure)):
            genom = {x: 100 for x in synapses}
            layer = Layer(
                    timer=t,
                    neuron_number=self.structure[layer_num],
                    layer_number=layer_num + 1,
                    genom=genom,
                    **kwargs
            )
            synapses = layer.get_synapses()
            self.layers.append(layer)

    def reset(self):
        self.feed.reset()

    def cycle_through(self):
        self.timer.reset()
        for o in self:
            pass

    def update_clock(self, t):
        self.clock = t

    def set_feed(self, datafeed):
        self.feed = datafeed

    def __next__(self):
        self.raw_data = self.feed.read()
        input_data, t = parse_aes(self.raw_data)  # cutting bits 0:22
        self.timer.set_time(t)  # устанавливаем часы по битам 22:0 входного пакета данных
        for layer in self.layers:
            layer.tick(input_data)
            input_data = layer.get_fired_synapses()
        self.frame += 1
        return self.layers[-1].get_fired_neurons()

    def __iter__(self):
        return self

    def get_journals_from_layer(self, layer=-1, output=True):
        if output:
            return self.layers[layer].get_output_journals()
        if not output:
            return self.layers[layer].get_input_journals()

    def calculate_fitness(self):
        R = np.array(self.get_journals_from_layer())
        C = np.array([[1 if (Defaults.answers[j//448] == i) else 1j for j in range(4912)] for i in range(1, 9)])
        C = C.transpose()
        F = np.matmul(R, C)
        score = 0
        cols = list(range(F.shape[1]))
        rows = list(range(F.shape[0]))
        shuffle(rows)
        preferred_order = [0]*8
        for i in rows:
            m = 0
            n = cols[0]
            for j in cols:
                if np.real(F[i][j]) > m:
                    m = F[i][j]
                    n = j
            score += m
            cols.remove(n)
            preferred_order[i] = n
        self.layers[-1].rearrange_neurons(preferred_order)
        score = (-100/np.log(np.real(score))+100)*(
                    np.real(score)/(
                        np.real(score)+np.imag(score)
                    )
                )
        return score

    def get_genom(self):
        return [layer.get_genom() for layer in self.layers]

    def random_genom(self):
        return [layer.random_genom() for layer in self.layers]

    def set_genom(self, genom):
        for l_num in range(len(self.layers)):
            self.layers[l_num].set_genom(genom[l_num])

    def mutate(self, genom1=False, genom2=False):
        if not genom1 and not genom2:
            self.set_genom(self.random_genom())
        elif not genom2:
            genom2 = genom1
        else:
            for l_num in range(len(self.layers)):
                self.layers[l_num].mutate(genom1[l_num], genom2[l_num])


def save_output_to_file(indicator, journals):
    xdata, ydata = [], []
    xdata = range(len(journals[0]))
    for data in journals:
        ydata.append([k*ans_amp for k in data])
    ydata.append(ans)
    ydata = [[ydata[j][i] for j in range(len(ydata))] for i in range(len(ydata[0]))]
    plt.plot(xdata, ydata)
    plt.savefig(str(indicator)+'.png')
    plt.close()

if __name__ == "__main__":
    timer = Timer(0)
    feed_opts = {'mode': 'file', 'source': 'out.bin'}
    feed = CameraFeed(**feed_opts)
    model = Network(t=timer, datafeed=feed, structure=(28*28, 8), wta=1, learn=True)
    best_scores = []
    best_two_genoms = [False, False]
    ans = [Defaults.answers[i// 498] for i in range(4912)]
    ans_amp = max(ans)
    folder = f"exp{int(time.time())}"
    mkdir(folder)

    for i in range(50):
        genoms = []
        scores = []
        journals = []
        print(f"=====generation {i}=====")
        for j in range(7):
            starttime = time.time()

            model.mutate(*best_two_genoms)

            model.cycle_through()

            fitness = model.calculate_fitness()
            print(time.time() - starttime, fitness)
            scores.append(fitness)
            genoms.append(model.get_genom())

            journals.append(model.get_journals_from_layer())

            timer.reset()

        s_scores = list(reversed(sorted(scores)))
        best_scores.append(s_scores[0])
        index1 = scores.index(s_scores[0])
        index2 = scores.index(s_scores[1])
        best_two_genoms = (genoms[index1], genoms[index2])
        save_output_to_file(f"{folder}/stage{i}fit{s_scores[0]}", journals[index1])
        save_output_to_file(f"{folder}/stage{i}fit{s_scores[1]}", journals[index2])

    plt.plot(list(range(len(best_scores))), best_scores)
    plt.show()
    plt.plot(list(range(len(best_scores))), best_scores)
    plt.savefig(f"{folder}/graph.png")
    pickle.dump(best_two_genoms[0], open(f"{folder}/f{s_scores[0]}.genom", "wb"))
