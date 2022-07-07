import pickle
import time
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Type

import matplotlib.pyplot as plt
import cv2
import numpy as np
import configparser as cm
import AERGen as ag
import random
import os
import csv
import sqlite3 as sq3
from concurrent import futures
from utility import *

#random.seed(42)
test_set = ["traces/b-t-500.bin",
            "traces/bl-tr-500.bin",
            "traces/br-tl-500.bin",
            "traces/l-r-500.bin",
            "traces/r-l-500.bin",
            "traces/t-b-500.bin",
            "traces/tl-br-500.bin",
            "traces/tr-bl-500.bin"
            ]
TEST_SETS = test_set * 5
random.shuffle(TEST_SETS)


class DataFeed:
    def __init__(self, feed_type, model):
        self.buf = []
        self.feed_type = feed_type
        self.source = ""
        self.terminate = False
        self.time_offset = 0
        self.model = model

    def load(self, source, data=None):
        self.source = source
        self.terminate = False
        if self.feed_type == "iter":
            self.buf = data
        if self.feed_type == "stream":
            self.connect_stream()

    def next_events(self, peek=False):
        events = []
        if self.feed_type == "iter":
            if len(self.buf) == 1:
                if not peek:
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
            if not peek:
                self.buf = self.buf[i:]
            return events
        if self.feed_type == "stream":
            events = self.await_event()
        return events

    def connect_stream(self):
        pass

    def await_event(self):
        pass


NeuronParametersSet: Type["NeuronParametersSet"] = namedtuple("NeuronParametersSet", ["i_thres",
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

GeneralParametersSet: Type["GeneralParametersSet"] = namedtuple("GeneralParametersSet", ["inhibit_radius",
                                                           "epoch_length",
                                                           "execution_thres",
                                                           "terminate_on_epoch",
                                                           "wta",
                                                           "false_positive_thres",
                                                           "mask"
                                                           ])

LayerStruct = namedtuple("LayerStruct", ["neurons", "shape", "per_field_shape"])


@dataclass
class Model:
    neuron_parameters_set: NeuronParametersSet
    general_parameters_set: GeneralParametersSet
    state: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    time: int = 0
    logs: dict = field(default_factory=dict)
    pat_logs: list = field(default_factory=list)
    layers: list = field(default_factory=list)
    label: str = field(default_factory=str)


class Neuron:
    def __init__(self, model, output_address, inputs, learn=True, weights=None, mask=None, neuron_parameters_set=None):
        self.model = model
        if output_address not in model.logs:
            model.logs[output_address] = []
        if neuron_parameters_set is None:
            self.param_set = model.neuron_parameters_set
        else:
            self.param_set = neuron_parameters_set
        self.error = -1
        self.output_address = output_address
        self.learn = learn
        self.inputs = inputs
        self.weights = self.random_weights() if weights is None else weights
        self.weights_mask = {}
        if mask is not None:
            for x, row in enumerate(mask):
                for y, el in enumerate(row):
                    self.weights_mask[f"{format(x, '02x')}{format(y, '02x')}0"] = el * self.param_set.w_max
                    self.weights_mask[f"{format(x, '02x')}{format(y, '02x')}8"] = el * self.param_set.w_max

        else:
            self.weights_mask = {k: self.param_set.w_max for k in self.weights.keys()}
        self.input_level = 0
        self.output_level = 0
        self.t_last_spike = -1
        self.t_spike = -1
        self.inhibited_by = -1
        self.times_fired = 0
        self.ltp_times = {}
        self.label = ""
        self.log = []


    def copy(self, model):
        return Neuron(model, output_address=self.output_address, inputs=self.inputs, learn=self.learn)

    def update(self):
        self.output_level = 0
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.model.time
        if self.model.time <= self.inhibited_by:
            return 0

        state = self.model.state  # копируем для ускорения доступа
        inputs = self.inputs

        events = [(state[address], address) for address in inputs if state[address]]
        if not events:
            return False
        for event, address in events:
            self.t_last_spike, self.t_spike = self.t_spike, self.model.time
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.param_set.t_leak)
            self.input_level += self.weights[address]
            self.ltp_times[address] = self.model.time + self.param_set.t_ltp
        if self.param_set.activation_function == "DeltaFunction":
            self.output_level = int(self.input_level > self.param_set.i_thres)
        if self.output_level:
            self.times_fired += 1
            self.model.logs[self.output_address].append(self.model.time)
            min_level = self.param_set.w_min
            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                rest = [k for k in self.inputs if k not in not_rotten]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc
                    if self.weights[synapse] > self.weights_mask[synapse]:
                        self.weights[synapse] = self.weights_mask[synapse]
                for synapse in rest:
                    self.weights[synapse] -= self.param_set.a_dec
                    if self.weights[synapse] < min_level:
                        self.weights[synapse] = min_level
            self.ltp_times = {}
        return self.output_level

    def inhibit(self):
        if self.inhibited_by <= self.model.time:
            self.inhibited_by = self.model.time + self.param_set.t_inhibit

    def reset(self, soft=False):
        if not soft:
            self.weights = self.random_weights()
            self.label = ""
            self.learn = True
        self.input_level = 0
        self.output_level = 0
        self.t_last_spike = -1
        self.t_spike = -1
        self.inhibited_by = -1
        self.times_fired = 0
        self.ltp_times = {}
        self.age = 0

    def attention_map(self):
        pixels_on = Defaults.pixels[1::2]
        pixels_off = Defaults.pixels[::2]
        all_pixels = [self.weights[on] if self.weights[on] > self.weights[off]
                      else -1 * self.weights[off]
                      for on, off in zip(pixels_on, pixels_off)]
        return self.output_address, \
               self.label, \
               self.error, \
               np.array(all_pixels).reshape((self.model.layers[-1].per_field_shape[0],
                                             self.model.layers[-1].per_field_shape[1])).transpose()

    def set_weights(self, weights):
        self.weights = weights.copy()

    def random_weights(self):
        return {i: random.random() * self.param_set.w_random * (self.param_set.w_max - self.param_set.w_min) for i in
                self.inputs}

def assess_error(neuron):
    """pat_log: матрица правильных ответов. Каждая строка отвечает 1 предъявленному паттерну, и имеет 1 в столбце
    соответствующем правильному паттерну, и 0 в остальных столбцах
    pat_key: список отметок времени в которые происходила
    смена паттерна
    pat_appearance: счёт предъявлений каждого паттерна
    self_log: вектор-строка срабатываний нейрона. Каждое значение - время выдачи спайка
    log_processed: сработал ли нейрон во время предъявления каждого паттерна
    ans_by_pat: вектор-строка верных и ложных срабатываний на каждый паттерн. Реальная часть числа в столбце i -
    кол-во верных срабатываний на паттерн i.
    ans_by_learned: вектор для оценки распознавания каждого паттерна нейроном. Реальная часть числа в столбце i -
    кол-во верных срабатываний на паттерн i, мнимая - количество всех остальных срабатываний"""
    log_processed = []
    pat_log_processed = []
    log = neuron.model.logs[neuron.output_address]
    pat_log = neuron.model.pat_logs
    patterns = np.array(list(set(list(zip(*pat_log))[0])))

    for index, (_, tstamp) in enumerate(pat_log[1:]):
        pat = pat_log[index][0]
        pat_log_processed.append(np.where(patterns == pat, 1, 0))
        a = 0
        i = 0
        for i, spike in enumerate(log):
            if spike > tstamp:
                break
            a = 1
        log = log[i:]
        log_processed.append(a)

    tstamp = pat_log[-1][1]
    pat = pat_log[-1][0]
    pat_log_processed.append(np.where(patterns == pat, 1, 0))
    a = 0
    i = 0
    for i, spike in enumerate(log):
        if spike > tstamp:
            a = 1
            break
    log_processed.append(a)

    log_processed, pat_log_processed = np.array(log_processed), np.array(pat_log_processed)

    pat_appearance = [np.count_nonzero(r) for r in np.transpose(pat_log_processed)]
    ans_by_pat = np.matmul(log_processed, pat_log_processed)
    ans_by_learned = [a+1j*(sum(log_processed)-a) for a in ans_by_pat]
    assess_error = lambda ans, pat_app, tot_pat_app: (pat_app - np.real(ans)) / pat_app + np.imag(ans) / \
                                                     (tot_pat_app - pat_app)
    neuron.weighted_error = {label: assess_error(ans, pat_app, sum(pat_appearance))
                             for label, ans, pat_app in zip(patterns, ans_by_learned, pat_appearance)}
    best_guess = sorted(neuron.weighted_error.items(), key=lambda x: x[1])[0]
    neuron.label = best_guess[0]
    neuron.error = best_guess[1]

    return {neuron.output_address: best_guess}


def construct_network(feed_type, source_file, learn=True, update_neuron_parameters={}, update_general_parameters={}):
    config = cm.ConfigParser()
    with open(source_file, 'r') as f:
        config.read_file(f)
    config["NEURON PARAMETERS"].update(update_neuron_parameters)
    config["GENERAL PARAMETERS"].update(update_general_parameters)
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
    gps = GeneralParametersSet(int(config["GENERAL PARAMETERS"]["inhibit_radius"]),
                               int(config["GENERAL PARAMETERS"]["epoch_length"]),
                               int(config["GENERAL PARAMETERS"]["execution_thres"]),
                               int(config["GENERAL PARAMETERS"]["terminate_on_epoch"]),
                               int(config["GENERAL PARAMETERS"]["wta"]),
                               float(config["GENERAL PARAMETERS"]["false_positive_thres"]),
                               config["GENERAL PARAMETERS"]["mask"])
    model = Model(neuron_parameters_set=nps, general_parameters_set=gps)
    if gps.mask != "none":
        mask = load_mask(gps.mask)
    else:
        mask = None
    structure = config["GENERAL PARAMETERS"]["structure"]
    if isinstance(structure, str):
        structure = [structure]
    layer = ""
    for layer in structure:
        model.state.update({s: 0 for s in config[layer]["inputs"].split(' ')})
        if config[layer]["type"] == "layer":
            model.layers.append(LayerStruct([Neuron(model, output, config[layer]["inputs"].split(' '), learn, mask=mask)
                                             for output in config[layer]["outputs"].split(' ')],
                                            tuple(map(int, config[layer]["shape"].split(' '))),
                                            tuple(map(int, config[layer]["per_field_shape"].split(' ')))))
    model.outputs = config[layer]["outputs"].split(' ')
    model.state.update({s: 0 for s in model.outputs})
    model.logs = {s: [] for s in model.state}
    return model, DataFeed(feed_type, model)

def label_neurons(model):
    result = {}
    for neuron in model.layers[-1].neurons:
        result.update(assess_error(neuron))
    return result


def load_mask(source):
    im = cv2.imread(source)
    bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    return np.around(np.divide(bw, 255.0), decimals=3)


def load_network(source):
    with open(source, "rb") as f:
        return pickle.load(f)


def next_training_cycle(donor_model, feed):
    donor_model.state = {k: 0 for k in donor_model.state.keys()}
    next_ev = feed.next_events()
    if feed.terminate:
        for layer in donor_model.layers:
            for neuron in layer.neurons:
                neuron.age += 1
    feed_events(donor_model, feed.source, next_ev)
    return not feed.terminate


def next_recognition_cycle(model, feed):
    model.state = {k: 0 for k in model.state.keys()}
    next_ev = feed.next_events()
    feed_events(model, feed.source, next_ev)
    return not feed.terminate


def feed_events(model, source, events):
    if events:
        model.time = events[0].time
    for ev in events:
        model.state[ev.address] = 1

    for layer in model.layers:
        layer_update(model, layer)


def layer_update(model, layer):
    [neuron.update() for neuron in layer.neurons]

    for row in range(layer.shape[0]):
        for col in range(layer.shape[1]):
            if layer.neurons[row * layer.shape[1] + col].output_level:
                radius = model.general_parameters_set.inhibit_radius
                for i in range(-radius, radius + 1):
                    for j in range(-radius + abs(i), radius - abs(i) + 1):
                        if layer.shape[0] > row + i >= 0 and layer.shape[1] > col + j >= 0:
                            layer.neurons[(row + i) * layer.shape[1] + col + j].inhibit()
                            if model.general_parameters_set.wta:
                                layer.neurons[(row + i) * layer.shape[1] + col + j].input_level = 0


def feed_test_trace_set(model, feed, test_set):
    datasets = []
    reset(model, feed, issoft=True)
    for neuron in np.array([layer.neurons for layer in model.layers]).flatten():
        neuron.learn = False
    for path in test_set:
        with open(path, 'r') as f:
            datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
    for n in datasets:
        alias, dataset = n
        feed.load(alias, dataset)
        while next_recognition_cycle(model, feed):
            pass


def select_weights_from_pool(model, layers_pool):
    neuron_pool = {}
    for neuron in np.array(layers_pool).flatten():
        if neuron.label:
            if neuron.label not in neuron_pool:
                neuron_pool[neuron.label] = []
            neuron_pool[neuron.label].append((neuron, neuron.error))

    neurons = {k: sorted(v, key=lambda x: x[1])[:model.general_parameters_set.execution_thres]
               for k, v in neuron_pool.items()}

    return neurons


def fill_model_from_pool(model: Model, training_pool: [Model]):
    for i, layer in enumerate(model.layers):
        layers_pool = [m.layers[i].neurons for m in training_pool]
        learnt = select_weights_from_pool(model, layers_pool)
        if learnt:
            free_neurons = [neuron for neuron in layer.neurons if not neuron.label]

            select = []

            for i in range(max(list(map(len, list(learnt.items()))))):
                for label, neurons in learnt.items():
                    if neurons:
                        select.append(neurons.pop())

            for recipient, (graft, error) in zip(free_neurons, select):
                recipient.set_weights(graft.weights)
                recipient.error = error
                recipient.label = graft.label

            already_there = set([neuron.label for neuron in layer.neurons if neuron.label])
            already_there_count = {k: [n.label for n in layer.neurons].count(k) for k in already_there}

    return already_there_count


def delete_duplicate_neurons(model):
    neurons = {}
    for neuron in model.layers[-1].neurons:
        if neuron.label:
            if neuron.label not in neurons:
                neurons[neuron.label] = []
            neurons[neuron.label].append((neuron, neuron.error))
    neurons = {k: sorted(v, key=lambda x: x[1])[:-model.general_parameters_set.execution_thres]
               for k,v in neurons.items()}
    trash = [_[0] for sub in neurons.values() for _ in sub ]
    for ne in trash:
        ne.reset()
        ne.learn = True
    return {k:len(v) for k,v in neurons.items()}

def glue_attention_maps(model, folder):
    res = np.ndarray([model.layers[-1].per_field_shape[0] * model.layers[-1].shape[0], 0])
    for y in range(model.layers[-1].shape[1]):
        row = np.ndarray([0, model.layers[-1].per_field_shape[1]])
        for x in range(model.layers[-1].shape[0]):
            synapse, label, map = model.layers[-1].neurons[y * model.layers[-1].shape[0] + x].attention_map()
            row = np.concatenate((row, map))
        res = np.concatenate((res, row), axis=1)
    plt.close()
    plt.imshow(res)
    plt.savefig(f"{folder}/attention_maps.png")


def save_attention_maps(model, folder: str):
    attention_maps = []
    if not os.path.exists(folder):
        os.mkdir(folder)
    for n in model.layers[-1].neurons:
        attention_maps.append(n.attention_map())

    for address, label, error, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.savefig(f"{folder}/att_map_{address}_{label}_err_{error:.3f}.png")


def show_attention_maps(model, folder: str):
    attention_maps = []
    if not os.path.exists(folder):
        os.mkdir(folder)
    for n in model.layers[-1].neurons:
        attention_maps.append(n.attention_map())

    for address, label, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.show()


def reset(model, feed=None, issoft=False):
    model.time = 0
    if feed:
        feed.time_offset = 0
    model.state = {s: 0 for s in model.state}
    model.pat_logs = []
    model.logs = {s: [] for s in model.state}
    for layer in model.layers:
        for neuron in layer.neurons:
            neuron.reset(issoft)

def announce_pattern(model, source):
    model.pat_logs.append((source, model.time))