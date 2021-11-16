from layer import Layer
from utility import *
from camera_feed import CameraFeed
from timer import *

import time
import matplotlib.pyplot as plt
import numpy as np
import pickle
import random
from os import mkdir

#   проблемы
#   в выходных журналах нейронов больше записей, чем прошло импульсов
#
#

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
        self.frame = 0

    def cycle_through(self):
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

    def calculate_fitness(self, answers):
        R = np.array(self.get_journals_from_layer())
        C = np.array(answers)
        C = C.transpose()
        F = np.matmul(R, C)
        score = 0
        cols = list(range(F.shape[1]))
        rows = list(range(F.shape[0]))
        random.shuffle(rows)
        preferred_order = [0]*number_of_categories
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

    def mutate(self, *args):
        if len(args) == 0:
            self.set_genom(self.random_genom())
        else:
            for l_num in range(len(self.layers)):
                self.layers[l_num].mutate(*[genom[l_num] for genom in args])


def save_output_to_file(indicator, journals):
    xdata, ydata = [], []
    xdata = range(len(journals[0]))
    for data in journals:
        ydata.append([k for k in data])
    ydata = [[ydata[j][i] for j in range(len(ydata))] for i in range(len(ydata[0]))]
    plt.plot(xdata, ydata)
    plt.savefig(str(indicator)+'.png')
    plt.close()

if __name__ == "__main__":

    input_layer = 28*28
    output_layer = 8
    number_of_categories = 8

    generation_num = 10
    specimen_num = 5
    traces_num = 200

    timer = Timer(0)
    feed_opts = {'mode': 'file', 'source': 'trace_up.bin'}
    feed = CameraFeed(**feed_opts)

    model = Network(t=timer, datafeed=feed, structure=(input_layer, output_layer), wta=True, learn=True)
    best_scores = []
    best_genoms = []
    traces = []
    folder = f"exp{int(time.time())}"
    mkdir(folder)

    for i in range(generation_num):
        # number of generations
        genoms = []
        scores = []
        journals = []
        print(f"=====generation {i}=====")
        for j in range(specimen_num):
            #number of specimen in generation
            answers = [[], [], [], [], [], [], [], []]
            traces = [random.choice(Defaults.files) for _ in range(traces_num)]
            starttime = time.time()
            model.mutate(*best_genoms)
            tot = 0
            timer.reset()
            for n in range(traces_num):
                #number of traces to train with
                feed.load(traces[n])
                model.cycle_through()

                index_of_trace = Defaults.files.index(traces[n])
                answers[index_of_trace].append([tot, model.frame])
                tot += model.frame

                model.reset()

            genoms.append(model.get_genom())
            journals.append(model.get_journals_from_layer())

            ans = np.zeros((number_of_categories, tot))

            for i in range(ans.shape[0]):
                #create matrix of answers
                row = np.zeros(tot)
                for entry in answers[i]:
                    row += np.hstack((np.zeros(entry[0]),
                                      np.ones(entry[1]),
                                      np.zeros(tot-sum(entry))
                                      ))
                ans[i] = row

            fitness = model.calculate_fitness(ans)
            scores.append(fitness)
            print(time.time() - starttime, fitness)

        s_scores = list(reversed(sorted(scores)))
        best_scores.append(s_scores[0])
        index1 = scores.index(s_scores[0])
        index2 = scores.index(s_scores[1])
        best_genoms = (genoms[index1], genoms[index2])

    plt.plot(list(range(len(best_scores))), best_scores)
    plt.show()
    plt.plot(list(range(len(best_scores))), best_scores)
    plt.savefig(f"{folder}/graph.png")
    pickle.dump(best_genoms[0], open(f"{folder}/f{s_scores[0]}.genom", "wb"))
