from main import *
from utility import *
import time
import numpy as np
import pickle
import matplotlib.pyplot as plt

def recognize():
    weights = pickle.load(open('exp1638541627\\f.weights', 'rb'))
    model = init_model(model_options = {'learn': False, 'structure': [7,]})
    #target_model.set_weights(weights)
    model.set_random_weights()
    model.feed.load('resources\\out.bin')
    [model.logger.add_watch(neuron, 'input_level') for neuron in model.layers[0].neurons]
    while frame := model.next():
        pass

    input_levels = np.array([model.logger.get_logs(neuron, 'input_level') for neuron in model.layers[0].neurons])
    output_levels = np.array([model.logger.get_logs(neuron, 'output_level') for neuron in model.layers[0].neurons])
    for i in input_levels:
        plt.plot(list(range(input_levels.shape[1])), i)
        plt.show()
    for o in output_levels:
        plt.plot(list(range(input_levels.shape[1])), o)
        plt.show()
    #plt.plot(list(range(input_levels.shape[1])), input_levels.transpose())
    #plt.show()
    #plt.plot(list(range(input_levels.shape[1])), output_levels.transpose())
    #plt.show()
recognize()