from model import *
import pickle
import numpy as np
import matplotlib.pyplot as plt
import time


class Feed:
    tot = 100

    def __init__(self, t):
        self.f = self.tot
        self.timer = t

    def parse_aer(self, data):
        return [data, ], self.timer.clock

    def get_pixels(self):
        return ["0101",]

    def __iter__(self):
        return self

    def __next__(self):
        self.f -= 1
        if self.f == 0:
            raise StopIteration
        self.timer.clock = (self.tot - self.f) * 100
        return "0101"

total_time = 0

def run():
    model_opts = {'learn': False,
                  'structure': [1, ],
                  'datafeed': Feed(Timer.get_instance()),
                  'parameter_set':  {
                                        'i_thres': 1000,
                                        't_ltp': 1 * 10 ** 3,
                                        't_refrac': 0.5 * 10 ** 3,
                                        't_inhibit': 10 * 10 ** 2,
                                        't_leak': 3 * 10 ** 3,
                                        'w_min': 1,
                                        'w_max': 100,
                                        'a_dec': 5,
                                        'a_inc': 100,
                                        'randmut': 0.001
                                    }
                  }
    model = SpikeNetwork(**model_opts)
    model.set_random_weights()
    number_of_categories = 1
    total_time = 0
    o = []
    while frame := model.next():
        starttime = time.time_ns()
        total_time += time.time_ns() - starttime
        o.append([])
        o[-1].append(model.layers[0].neurons[0].output_level)
        o[-1].append(model.layers[0].neurons[0].input_level/model_opts['parameter_set']['i_thres'])
    return o

plt.plot(list(range(Feed.tot-1)), run())
plt.show()
print(total_time * 10 ** -9)