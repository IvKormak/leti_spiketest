from dataclasses import dataclass
from collections import namedtuple
from typing import Union, Tuple, Dict, List, Type, Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import configparser as cm
import random
import pickle

from utility import *


class DataFeed:
    def __init__(self, source, type):
        self.buf = ""
        self.type = type
        self.load(source)
        self.terminate = False

    def load(self, source):
        if self.type == "file":
            with open(source, "r") as file:
                hexadecimals = "0123456789abcdef"
                while char := file.read(1):
                    if char in hexadecimals:
                        self.buf += char
        if self.type == "stream":
            self.connect_stream()

    def next_events(self):
        ev = []
        if self.type == "file":
            rest = len(self.buf)
            if rest < 20:
                self.terminate = True
                return [parse_aer(self.buf[0:10])]
            i = 1
            ev = [parse_aer(self.buf[0:10])]
            time = parse_aer(self.buf[0:10])[1]
            while time == parse_aer(self.buf[i*10:(i+1)*10])[1]:
                ev.append(parse_aer(self.buf[i*10:(i+1)*10]))
                i += 1
                if rest < i*10:
                    break
            self.buf = self.buf[i*10:]
        if self.type == "stream":
            ev = self.await_event()
        return ev

    def connect_stream(self):
        pass

    def await_event(self):
        pass


Event = namedtuple("Event", ["address", "time"])

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
        self.weights = self.set_random_weights() if not weights else weights

        self.input_level = 0
        self.output_level = 0
        self.t_last_spike = -1
        self.t_spike = -1
        self.inhibited_by = -1
        self.times_fired = 0
        self.ltp_times = {}
        self.label = ""

    def update(self):
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.model.time
        if self.model.time <= self.inhibited_by:
            return False
        self.output_level = 0
        events = [self.model.state[address] for address in self.inputs]
        for address in self.inputs:
            self.model.state[address] = 0
        if not events:
            return False
        for event in events:
            self.t_last_spike, self.t_spike = self.t_spike, self.model.time
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.param_set.t_leak)
            if self.weights[event.address] > self.param_set.w_max + self.times_fired*self.param_set.a_dec:
                self.input_level += self.param_set.w_max
            elif self.weights[event.address] < self.param_set.w_min + self.times_fired*self.param_set.a_dec
                self.input_level += self.param_set.w_min
            else:
                self.input_level += self.weights[event.address] - self.times_fired * self.param_set.a_dec
            self.ltp_times[event.address] = self.model.time + self.param_set.t_ltp
        self.output_level = self.param_set.activation_function(self.input_level)
        if self.output_level:
            self.times_fired += 1
            self.input_level = 0
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc + self.param_set.a_dec
                self.ltp_times = {}
        return self.output_level

    def inhibit(self):
        if self.inhibited_by > self.model.time:
            self.inhibited_by += self.param_set.t_inhibit
        else:
            self.inhibited_by = self.model.time + self.param_set.t_inhibit

    def reset(self):
        pass

    def set_weights(self, weights):
        self.weights = weights.copy()

    def set_random_weights(self):
        self.weights = {i: random.random() * self.param_set.w_random * (self.param_set.w_max - self.param_set.w_min) for i in self.inputs}


def construct_network(learn=True):
    config = cm.ConfigParser()
    file = "network1.txt"
    with open(file, 'r') as f:
        config.read_file(f)
    nps = NeuronParametersSet(config["NEURON PARAMETERS"]["i_thres"],
                              config["NEURON PARAMETERS"]["t_ltp"],
                              config["NEURON PARAMETERS"]["t_refrac"],
                              config["NEURON PARAMETERS"]["t_inhibit"],
                              config["NEURON PARAMETERS"]["t_leak"],
                              config["NEURON PARAMETERS"]["w_min"],
                              config["NEURON PARAMETERS"]["w_max"],
                              config["NEURON PARAMETERS"]["w_random"],
                              config["NEURON PARAMETERS"]["a_dec"],
                              config["NEURON PARAMETERS"]["a_inc"],
                              config["NEURON PARAMETERS"]["activation_function"])
    lps = LayerParametersSet(config["NEURON PARAMETERS"]["inhib_radius"])
    model = Model(state={}, time=0, logs=[], layers=[], neuron_parameters_set=nps, layer_parameters_set=lps)
    structure = config["NEURON PARAMETERS"]["structure"]
    layer = ""
    for layer in structure:
        model.state.update({s: 0 for s in config[layer]["inputs"]})
        if type == "perceptron_layer":
            model.layers.append({'neurons': [Neuron(model, output, config[layer]["inputs"], learn)
                                             for output in config[layer]["outputs"]]})
            model.layers[-1].update({'shape': config[layer]["shape"]})
    model.outputs = config[layer]["outputs"]
    model.state.update({s: 0 for s in model.outputs})


def layer_update(model, layer):
    [neuron.update() for neuron in layer["neurons"]]

    for row in range(layer["shape"][0]):
        for col in range(layer["shape"][1]):
            if layer["neurons"][row * layer["shape"][1] + col].output_level:
                radius = model.LayerParametersSet.inhibit_radius
                for i in range(-radius, radius + 1):
                    for j in range(-radius + abs(i), radius - abs(i) + 1):
                        if layer["shape"][0] > row + i >= 0 and layer["shape"][1] > col + j >= 0:
                            layer["neurons"][(row + i) * layer["shape"][1] + col + j].inhibit()



def loop(model, feed_source, feed_type):
    feed = DataFeed(feed_source, feed_type)
    while not feed.terminate:
        next_ev = feed.next_events()
        model.time = next_ev[0][1]
        for synapse, time in next_ev:
            model.state[synapse] = 1

        for layer in model.layers:
            layer_update(model, layer)

        for o in model.outputs:
            if model.state[o]:
                model.logs.append((o, feed_source))
                model.state[o] = 0

    label_neurons(model)
    save_attention_maps(model, "/")

def label_neurons(model):

    neuron_journals = [l[0] for l in model.logs]
    teacher_journals = [l[1] for l in model.logs]
    all_traces = list(set(teacher_journals))
    all_neurons = list(set(neuron_journals))

    R = [[] for _ in all_neurons]
    C = teacher_journals

    for fired_neuron, right_answer in model.logs:
        for n in range(len(R)):
            R[n].append(all_neurons[n] == fired_neuron)

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
        rename_dict[all_neurons[neuron_number]] = all_traces[trace_best_guessed]

    for neuron in model.layers[-1]:
        neuron.label = rename_dict[neuron.output_address]

def parse_aer(raw_data: str) -> Tuple[str, int]:
    raw_data = int(raw_data, base=16)
    synapse = raw_data >> Defaults.time_bits
    synapse = synapse << 3
    synapse = format(synapse, '05x')
    time = raw_data & Defaults.time_mask
    return synapse, time


def save_attention_maps(model, folder: str):
    attention_maps = []
    pixels_on = Defaults.pixels[1::2]
    pixels_off = Defaults.pixels[::2]
    for n in model.layers[-1]:
        sum = [n.weights[on] if n.weights[on] > n.weights[off]
               else -1 * n.weights[off]
               for on, off in zip(pixels_on, pixels_off)]
        attention_maps.append((n.output_address, n.label, np.array(sum).reshape((28, 28)).transpose()))

    for address, label, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.savefig(f"{folder}/att_map_{address}_{label}.png")

def test(model):
    arrows = pickle.load(open('resources\\arrows.bin', 'rb'))

    white_square = np.multiply(np.ones((28, 28)), 255)
    current_frame = np.copy(white_square)
    current_arrow = np.copy(white_square)
    frames = []
    frames_shown = 0

    frame = model.next()
    while isinstance(frame, np.ndarray):
        if any(frame):
            frames_shown = 0
            current_arrow = arrows[frame[0]]
        if frames_shown > 50:
            current_arrow = np.copy(white_square)
        #data = model.raw_data
        synapse = parse_aer(data)[0][0]
        x_coord = int(synapse[0:2], base=16)
        y_coord = int(synapse[2:4], base=16)
        color = (synapse[4] == '0') * 255
        current_frame[y_coord][x_coord] = color
        frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
        frames_shown += 1
        #frame = model.next()

    iio.mimwrite('animation2.gif', frames, fps=60)
