from model import *
from utility import *
import time
import numpy as np
import pickle
from os import mkdir

def train():
    model_opts = {'learn': True,
                  'structure': [7,]}
    model = init_model(model_options = model_opts)
    #model.set_random_weights()
    traces_num = 500
    number_of_categories = 7
    traces = [random.choice(Defaults.files) for _ in range(traces_num)]
    print(traces)
    #traces = [Defaults.files[0] for _ in range(traces_num)]
    total_time = 0
    for trace in traces:
        model.feed.load(trace)
        while frame := model.next():
            starttime = time.time_ns()
            model.teacher.output = [1 if n == Defaults.files.index(trace) else 1j
                                    for n in range(number_of_categories)]
            total_time += time.time_ns() - starttime
    print(total_time*10**-9)
    fitness = model.calculate_fitness()
    print(fitness)
    folder = f"exp{int(time.time())}"
    mkdir(folder)
    model.save_attention_maps(folder)

    pickle.dump(model.get_weights(), open(f"{folder}\\f.weights", "wb"))

train()