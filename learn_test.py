from model import *
from utility import *
import time
import numpy as np
import pickle
from os import mkdir

def train():
    model = init_model(model_options={
                           'learn': True,
                           'structure': ((4,4),),
                           'decimation_coefficient': 0,
                           'inhib_radius': 2
                       },
                       feed_options={
                           'deprecate_timer':True,
                           'freq':5000
                       },
                       parameter_set = Experimental.param_set)
    model.set_random_weights()
    model.logger.add_watch(model.layers[0].neurons[0], 'input_level')
    traces_num = 200
    number_of_categories = 7
    traces = [random.choice(Defaults.files[:]) for _ in range(traces_num)]
    #traces = [Defaults.files[] for _ in range(traces_num)]
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

    i = model.logger.get_logs(model.layers[0].neurons[0], 'input_level')
    o = [v*10000 for v in model.logger.get_logs(model.layers[0].neurons[0], 'output_level')]
    t = model.logger.get_logs('timestamps')
    plt.plot(t,i,t,o)
    #plt.show()

    fitness = model.calculate_fitness()
    print(fitness)
    folder = f"exp{int(time.time())}"
    mkdir(folder)
    model.save_attention_maps(folder)

    pickle.dump(model.get_weights(), open(f"{folder}\\f.weights", "wb"))

train()