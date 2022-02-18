from dataclasses import dataclass
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import configparser as cm
import AERGen as ag
import random
import pickle
import os
import time
from utility import *


class DataFeed:
    def __init__(self, type, model):
        self.buf = []
        self.type = type
        self.source = ""
        self.terminate = False
        self.time_offset = 0
        self.model = model

    def load(self, source, data=None):
        self.source = source
        self.terminate = False
        if self.type == "iter":
            self.buf = data
        if self.type == "stream":
            self.connect_stream()

    def next_events(self):
        events = []
        if self.type == "iter":
            if len(self.buf) == 1:
                self.terminate = True
                self.time_offset += self.buf[0].time
                return self.buf
            time = self.buf[0].time
            i = 0
            for i in range(len(self.buf)):
                ev = self.buf[i]
                if ev.time > time:
                    break
                events.append(ag.Event(ev.address, ev.position, ev.polarity, time + self.time_offset))
            self.buf = self.buf[i:]
            return events
        if self.type == "stream":
            events = self.await_event()
        return events

    def connect_stream(self):
        pass

    def await_event(self):
        pass

NeuronParametersSet = namedtuple("NeuronParametersSet", ["i_thres",
                                                         "t_ltp",
                                                         "t_refrac",
                                                         "t_inhibit",
                                                         "t_leak",
                                                         "w_min",
                                                         "w_max",
                                                         "w_random",
                                                         "a_dec",
                                                         "a_inc",
                                                         "activation_function"
                                                         ])

LayerParametersSet = namedtuple("LayerParametersSet", ["inhibit_radius"])


@dataclass
class Model:
    state: {}
    outputs = []
    time: int
    logs: []
    layers: []
    neuron_parameters_set: NeuronParametersSet
    layer_parameters_set: LayerParametersSet


class Neuron:
    def __init__(self, model, output_address, inputs, learn=True, weights=None):
        self.model = model
        self.param_set = model.neuron_parameters_set
        self.output_address = output_address
        self.learn = learn
        self.inputs = inputs
        self.weights = self.random_weights() if not weights else weights

        self.input_level = 0
        self.output_level = 0
        self.t_last_spike = -1
        self.t_spike = -1
        self.inhibited_by = -1
        self.times_fired = 0
        self.ltp_times = {}
        self.label = ""

    def update(self):
        self.output_level = 0
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.model.time
        if self.model.time <= self.inhibited_by:
            return 0

        state = self.model.state #копируем для ускорения доступа
        inputs = self.inputs

        events = [(state[address], address) for address in self.inputs if state[address]]

        if not events:
            return False
        for event, address in events:
            self.t_last_spike, self.t_spike = self.t_spike, self.model.time
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.param_set.t_leak)
            self.input_level += self.weights[address] - self.times_fired * self.param_set.a_dec
            self.ltp_times[address] = self.model.time + self.param_set.t_ltp
        if self.param_set.activation_function == "DeltaFunction":
            self.output_level = int(self.input_level>self.param_set.i_thres)
        if self.output_level:

            min_level = self.param_set.w_min + self.times_fired*self.param_set.a_dec
            max_level = self.param_set.w_max + self.times_fired*self.param_set.a_dec

            self.times_fired += 1
            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                rest = [k for k in self.inputs if k not in not_rotten and
                        self.weights[k] < min_level]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc + self.param_set.a_dec
                    if self.weights[synapse] > max_level:
                        self.weights[synapse] = max_level
                for synapse in rest:
                    self.weights[synapse] = min_level
                self.ltp_times = {}
        return self.output_level

    def inhibit(self):
        if self.inhibited_by > self.model.time and self.output_level:
            self.inhibited_by += self.param_set.t_inhibit
        else:
            self.inhibited_by = self.model.time + self.param_set.t_inhibit

    def reset(self):
        self.weights = self.random_weights()
        self.input_level = 0
        self.output_level = 0
        self.t_last_spike = -1
        self.t_spike = -1
        self.inhibited_by = -1
        self.times_fired = 0
        self.ltp_times = {}
        self.label = ""

    def attention_map(self):
        pixels_on = Defaults.pixels[1::2]
        pixels_off = Defaults.pixels[::2]
        sum = [self.weights[on] if self.weights[on] > self.weights[off]
               else -1 * self.weights[off]
               for on, off in zip(pixels_on, pixels_off)]
        return self.output_address, self.label, np.array(sum).reshape((28, 28)).transpose()

    def set_weights(self, weights):
        self.weights = weights.copy()

    def random_weights(self):
        return {i: random.random() * self.param_set.w_random * (self.param_set.w_max - self.param_set.w_min) for i in self.inputs}


def construct_network(feed_type, file="network1.txt", learn=True):
    config = cm.ConfigParser()
    with open(file, 'r') as f:
        config.read_file(f)
    nps = NeuronParametersSet(int(config["NEURON PARAMETERS"]["i_thres"]),
                              int(config["NEURON PARAMETERS"]["t_ltp"]),
                              int(config["NEURON PARAMETERS"]["t_refrac"]),
                              int(config["NEURON PARAMETERS"]["t_inhibit"]),
                              int(config["NEURON PARAMETERS"]["t_leak"]),
                              int(config["NEURON PARAMETERS"]["w_min"]),
                              int(config["NEURON PARAMETERS"]["w_max"]),
                              float(config["NEURON PARAMETERS"]["w_random"]),
                              int(config["NEURON PARAMETERS"]["a_dec"]),
                              int(config["NEURON PARAMETERS"]["a_inc"]),
                              config["NEURON PARAMETERS"]["activation_function"])
    lps = LayerParametersSet(int(config["LAYER PARAMETERS"]["inhib_radius"]))
    model = Model(state={}, time=0, logs=[], layers=[], neuron_parameters_set=nps, layer_parameters_set=lps)
    structure = config["LAYER PARAMETERS"]["structure"]
    if isinstance(structure, str):
        structure = [structure]
    layer = ""
    for layer in structure:
        model.state.update({s: 0 for s in config[layer]["inputs"].split(' ')})
        if config[layer]["type"] == "perceptron_layer":
            model.layers.append({'neurons': [Neuron(model, output, config[layer]["inputs"].split(' '), learn)
                                             for output in config[layer]["outputs"].split(' ')],
                                 'shape': [int(s) for s in config[layer]["shape"].split(' ')]})
    model.outputs = config[layer]["outputs"].split(' ')
    model.state.update({s: 0 for s in model.outputs})
    feed = DataFeed(feed_type, model)
    return model, feed

def load_network(model, feed_type, learn=False):
    model.time = 0
    model.state = {s: 0 for s in model.state}
    model.logs = []
    for layer in model.layers:
        for neuron in layer['neurons']:
            neuron.learn = learn
    feed = DataFeed(feed_type, model)
    return model, feed

def reset_network(model, feed):
    model.time = 0
    model.state = {s: 0 for s in model.state}
    model.logs = []
    for layer in model.layers:
        for neuron in layer['neurons']:
            neuron.reset()
    feed.time_offset = 0

def layer_update(model, layer):
    [neuron.update() for neuron in layer["neurons"]]

    for row in range(layer["shape"][0]):
        for col in range(layer["shape"][1]):
            if layer["neurons"][row * layer["shape"][1] + col].output_level:
                radius = model.layer_parameters_set.inhibit_radius
                for i in range(-radius, radius + 1):
                    for j in range(-radius + abs(i), radius - abs(i) + 1):
                        if layer["shape"][0] > row + i >= 0 and layer["shape"][1] > col + j >= 0:
                            layer["neurons"][(row + i) * layer["shape"][1] + col + j].inhibit()


def next_network_cycle(model, feed):
    if feed.terminate:
        return False
    next_ev = feed.next_events()
    model.time = next_ev[0].time
    for ev in next_ev:
        model.state[ev.address] = 1

    for layer in model.layers:
        layer_update(model, layer)

    for o in model.outputs:
        if model.state[o]:
            model.logs.append((o, feed.source))

    for k in model.state.keys():
        model.state[k] = 0
    return True


def label_neurons(model):
    neuron_journals = [l[0] for l in model.logs]
    teacher_journals = [l[1] for l in model.logs]
    all_traces = list(set(teacher_journals))
    all_neurons = list(set(neuron_journals))

    R = [[] for _ in all_neurons]
    C = []

    for fired_neuron, right_answer in model.logs:
        C.append([0 if i != all_traces.index(right_answer) else 1 for i in range(len(R))])
        for n in range(len(R)):
            R[n].append(int(all_neurons[n] == fired_neuron))


    R, C = np.array(R), np.array(C)
    F = np.matmul(R, C)

    trace_numbers = list(range(C.shape[1]))
    neuron_numbers = list(range(R.shape[0]))

    rename_dict = {}

    for neuron_number in neuron_numbers:
        maximum_score_for_trace = 0
        if len(trace_numbers) == 0:
            break
        trace_best_guessed = trace_numbers[0]
        for trace_number in trace_numbers:
            if np.real(F[neuron_number][trace_number]) > maximum_score_for_trace:
                maximum_score_for_trace = F[neuron_number][trace_number]
                trace_best_guessed = trace_number
        rename_dict[all_neurons[neuron_number]] = os.path.basename(all_traces[trace_best_guessed])

    unique_labels = {l: list(rename_dict.values()).count(l) for l in set(rename_dict.values())}

    for neuron in model.layers[-1]['neurons']:
        neuron.label = rename_dict[neuron.output_address]

    return unique_labels

def save_attention_maps(model, folder: str):
    attention_maps = []
    if not os.path.exists(folder):
        os.mkdir(folder)
    for n in model.layers[-1]['neurons']:
        attention_maps.append(n.attention_map())

    for address, label, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.savefig(f"{folder}/att_map_{address}_{label}.png")

def test():
    arrows = pickle.load(open('resources\\arrows.bin', 'rb'))

    white_square = np.multiply(np.ones((28, 28)), 255)
    current_frame = np.copy(white_square)
    current_arrow = np.copy(white_square)
    frames = []
    frames_shown = 0

    model = construct_network()
    loop(model, "resources\\out.bin")

    frame = model.next()
    # переписать код используя AERGEN
    # возможно необходимо внедрить его внутрь loop или использовать Наблюдателя
    while isinstance(frame, np.ndarray):
        if any(frame):
            frames_shown = 0
            current_arrow = arrows[frame[0]]
        if frames_shown > 50:
            current_arrow = np.copy(white_square)
        # data = model.raw_data
        synapse = parse_aer(data)[0][0]
        x_coord = int(synapse[0:2], base=16)
        y_coord = int(synapse[2:4], base=16)
        color = (synapse[4] == '0') * 255
        current_frame[y_coord][x_coord] = color
        frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
        frames_shown += 1
        # frame = model.next()

    iio.mimwrite('animation2.gif', frames, fps=60)


if __name__ == "__main__":

    sets1000 = ["traces/b-t-1000.bin", "traces/l-r-1000.bin", "traces/r-l-1000.bin", "traces/t-b-1000.bin"]
    sets3000 = ["traces/b-t-3000.bin", "traces/l-r-3000.bin", "traces/r-l-3000.bin", "traces/t-b-3000.bin"]
    sets5000 = ["traces/b-t-5000.bin", "traces/l-r-5000.bin", "traces/r-l-5000.bin", "traces/t-b-5000.bin"]
    sets100 = ["traces/b-t-100.bin", "traces/l-r-100.bin", "traces/r-l-100.bin", "traces/t-b-100.bin"]
    sets500 = ["traces/b-t-500.bin", "traces/l-r-500.bin", "traces/r-l-500.bin", "traces/t-b-500.bin",
               "traces/bl-tr-500.bin", "traces/tl-br-500.bin", "traces/br-tl-500.bin", "traces/tr-bl-500.bin"]

    """sweep = [("resources/models/network2.txt", "0wr7")]"""
    sweep = [("resources/models/networkb.txt", "70inc60dec07wr1700ltp")]

    """model, feed = construct_network("iter", "resources/models/network2.txt")
    chosenSets = sets500
    datasets = []
    for path in chosenSets:
        with open(path, 'r') as f:
            datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
    for n in range(75):
        alias, dataset = random.choice(datasets)
        feed.load(alias, dataset)
        while next_network_cycle(model, feed):
            pass"""


    for file, folder in sweep:
        ts = time.time()
        model, feed = construct_network("iter", file)
        chosenSets = sets500
        datasets = []
        for path in chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
        div_n = 0
        for sampling in range(10):
            reset_network(model, feed)
            for n in range(75):
                alias, dataset = random.choice(datasets)
                feed.load(alias, dataset)
                while next_network_cycle(model, feed):
                    pass

            labels = label_neurons(model)
            div_n += len(labels)
            print(len(labels))
        div_n = div_n/10
        print(f"{folder}: learn {time.time() - ts} s")
        print(f"Diversity: {div_n}")

        save_attention_maps(model, f"experiments_results/{folder}")

        with open(f"experiments_results/{folder}/model.pkl", "wb") as f:
            pickle.dump(model, f)