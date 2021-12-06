from model import *
from utility import *
import time
import numpy as np
import pickle
from os import mkdir

def train():
    model_opts = {'learn': True,
                  'structure': ((4,4),)}
    model = init_model(model_options = model_opts)
    model.set_random_weights()
    traces_num = 500
    number_of_categories = 7
    #traces = [random.choice(Defaults.files) for _ in range(traces_num)]
    traces = [Defaults.files[_%3] for _ in range(traces_num)]
    total_time = 0
    starttime = time.time_ns()
    for trace in traces:
        model.feed.load(trace)
        frame = np.array([])
        while isinstance(frame, np.ndarray):
            frame = model.next()
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
    pickle.dump(model, open(f"{folder}\\f.model", "wb"))

train()