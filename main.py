import pickle
import time
from dataclasses import dataclass, field
from collections import namedtuple
import matplotlib.pyplot as plt
import cv2
import numpy as np
import configparser as cm
import AERGen as ag
import random
import os
from concurrent import futures
from utility import *


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

GeneralParametersSet = namedtuple("GeneralParametersSet", ["inhibit_radius",
                                                           "epoch_length",
                                                           "execution_thres",
                                                           "terminate_on_epoch",
                                                           "wta",
                                                           "false_positive_thres",
                                                           "valuable_logs_part"
                                                           ])

LayerStruct = namedtuple("LayerStruct", ["neurons", "shape", "per_field_shape"])


@dataclass
class Model:
    neuron_parameters_set: NeuronParametersSet
    general_parameters_set: GeneralParametersSet
    state: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    time: int = 0
    logs: list = field(default_factory=list)
    layers: list = field(default_factory=list)


class Neuron:
    def __init__(self, model, output_address, inputs, learn=True, weights=None, mask=None):
        self.model = model
        self.param_set = model.neuron_parameters_set
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

        self.age = 0

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

            min_level = self.param_set.w_min
            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn and self.age <= self.model.general_parameters_set.epoch_length:
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
        if self.inhibited_by > self.model.time and self.output_level:
            self.inhibited_by += self.param_set.t_inhibit
        else:
            self.inhibited_by = self.model.time + self.param_set.t_inhibit

    def reset(self, soft=False):
        if not soft:
            self.weights = self.random_weights()
            self.label = ""
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
               np.array(all_pixels).reshape((self.model.layers[-1].per_field_shape[0],
                                             self.model.layers[-1].per_field_shape[1])).transpose()

    def set_weights(self, weights):
        self.weights = weights.copy()

    def random_weights(self):
        return {i: random.random() * self.param_set.w_random * (self.param_set.w_max - self.param_set.w_min) for i in
                self.inputs}


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
                               float(config["GENERAL PARAMETERS"]["valuable_logs_part"]))
    model = Model(neuron_parameters_set=nps, general_parameters_set=gps)
    if config["NEURON PARAMETERS"]["mask"] != "none":
        mask = load_mask(config["NEURON PARAMETERS"]["mask"])
    else:
        mask = None
    structure = config["GENERAL PARAMETERS"]["structure"]
    if isinstance(structure, str):
        structure = [structure]
    layer = ""
    for layer in structure:
        model.state.update({s: 0 for s in config[layer]["inputs"].split(' ')})
        if config[layer]["type"] == "perceptron_layer":
            model.layers.append(LayerStruct([Neuron(model, output, config[layer]["inputs"].split(' '), learn, mask=mask)
                                             for output in config[layer]["outputs"].split(' ')],
                                            tuple(map(int, config[layer]["shape"].split(' '))),
                                            tuple(map(int, config[layer]["per_field_shape"].split(' ')))))
    model.outputs = config[layer]["outputs"].split(' ')
    model.state.update({s: 0 for s in model.outputs})
    return model, DataFeed(feed_type, model)


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
    model.time = events[0].time
    for ev in events:
        model.state[ev.address] = 1

    for layer in model.layers:
        layer_update(model, layer)

    for synapse in model.outputs:
        if model.state[synapse]:
            model.logs.append((synapse, source))


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


def label_neurons(donor_model):
    neuron_journals = [l[0] for l in donor_model.logs[int(len(
        donor_model.logs) * donor_model.general_parameters_set.valuable_logs_part):]]
    teacher_journals = [l[1] for l in donor_model.logs[int(len(
        donor_model.logs) * donor_model.general_parameters_set.valuable_logs_part):]]
    all_traces = list(set(teacher_journals))
    all_neurons = list(set(neuron_journals))

    R = [[] for _ in all_neurons]
    C = []

    for fired_neuron, right_answer in zip(neuron_journals, teacher_journals):
        C.append([1j if i != all_traces.index(right_answer) else 1 for i in range(len(R))])
        for n in range(len(R)):
            R[n].append(int(all_neurons[n] == fired_neuron))

    R, C = np.array(R), np.array(C)
    F = np.matmul(R, C)

    mean_category_apperance = len(C)/len(all_traces)
    # reckognition error after doi:10.1109/ijcnn.2017.7966336
    weighted_error = lambda x: (mean_category_apperance - np.real(x)) / mean_category_apperance + np.imag(x) / ((len(C) - 1) * mean_category_apperance)

    recognition_error = {all_neurons[i]: weighted_error(r.max()) for i, r in enumerate(F)}
    rename_dict = {all_neurons[i]: os.path.basename(all_traces[np.where(r == r.max())[0][0]]) for i, r in enumerate(F)
                   if recognition_error[all_neurons[i]] < donor_model.general_parameters_set.false_positive_thres}

    countlabels = {val: list(rename_dict.values()).count(val) for val in rename_dict.values()}

    for neuron in donor_model.layers[-1].neurons:
        if neuron.output_address in rename_dict:
            neuron.label = rename_dict[neuron.output_address]

    donor_model.logs = []
    return {'labels': countlabels, 'recognition_error': recognition_error}


def select_weights_from_pool(layers_pool, dup_thres):
    learnt_categories = {}
    neuron_pool = np.array(layers_pool).flatten()
    for n in neuron_pool:
        if n.label:
            if n.label in learnt_categories:
                if learnt_categories[n.label]['count'] < dup_thres:
                    learnt_categories[n.label]['ids'].append(id(n))
                    learnt_categories[n.label]['weights'].append(n.weights)
                    learnt_categories[n.label]['count'] += 1
            else:
                learnt_categories[n.label] = {}
                learnt_categories[n.label]['ids'] = [id(n), ]
                learnt_categories[n.label]['weights'] = [n.weights, ]
                learnt_categories[n.label]['count'] = 1
    return learnt_categories


def fill_model_from_pool(model: Model, training_pool: [Model]):
    for i, layer in enumerate(model.layers):
        layers_pool = [m.layers[i].neurons for m in training_pool]
        learnt = select_weights_from_pool(layers_pool, model.general_parameters_set.execution_thres)
        already_there = set([neuron.label for neuron in layer.neurons if neuron.label])
        already_there_count = {k: [n.label for n in layer.neurons].count(k) for k in already_there}

        for k in already_there:
            if k in learnt:
                learnt[k]['count'] -= already_there_count[k]

        select = []
        free_neurons = [neuron for neuron in layer.neurons if not neuron.label]

        for label, e in learnt.items():
            if e['count'] > 0:
                for i in range(e['count']):
                    select.append((e['weights'][i], label))

        for recipient, (graft, label) in zip(free_neurons, select):
            recipient.set_weights(graft)
            recipient.label = label

        already_there = set([neuron.label for neuron in layer.neurons if neuron.label])
        already_there_count = {k: [n.label for n in layer.neurons].count(k) for k in already_there}

    return already_there_count


def delete_duplicate_neurons(model, countlabels):
    for neuron in model.layers[-1].neurons:
        if neuron.label:
            if countlabels[neuron.label] > model.general_parameters_set.execution_thres:
                countlabels[neuron.label] -= 1
                neuron.reset()
                neuron.learn = True


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

    for address, label, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.savefig(f"{folder}/att_map_{address}_{label}.png")


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


def reset(model, feed, issoft=False):
    model.time = 0
    feed.time_offset = 0
    model.state = {s: 0 for s in model.state}
    model.logs = []
    for layer in model.layers:
        for neuron in layer.neurons:
            neuron.reset(issoft)


if __name__ == "__main__":
    def train(model_and_feed):
        model, feed, id = model_and_feed
        datasets = []
        chosenSets = ["traces/b-t-500.bin",
                      "traces/bl-tr-500.bin",
                      "traces/br-tl-500.bin",
                      "traces/l-r-500.bin",
                      "traces/r-l-500.bin",
                      "traces/t-b-500.bin",
                      "traces/tl-br-500.bin",
                      "traces/tr-bl-500.bin"
                      ]
        for path in chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))
        for n in range(model.general_parameters_set.epoch_length):
            alias, dataset = random.choice(datasets)
            feed.load(alias, dataset)
            while next_training_cycle(model, feed):
                pass
        r = label_neurons(model)
        return {'log': [id, r], 'model': model}


    model_file = "resources/models/network3_c.txt"

    general_variations = [{"wta": "0",
                           "mask": "none"
                           },
                          {"wta": "1",
                           "mask": "resources/mask-42.bmp"
                           },
                          {"wta": "0",
                           "mask": "resources/mask-42.bmp"
                           },
                          {"wta": "1",
                           "mask": "none"
                           }
                          ]

    neuron_variations = {"w_random": ["0.5", "0.7", "1", "1.5"],
                         "a_inc": ["70", "85", "100", "125", "150", "200"],
                         "a_dec": ["50", "30", "80"],
                         "t_ltp": ["2000", "500", "1500", "1750", "2500", "3000", "4000"],
                         "w_min": ["1", "10"],
                         "w_max": ["1000", "800", "900"],
                         "t_refrac": ["10000", "8000", "12000"],
                         "t_leak": ["5000", "4000", "6000"],
                         "i_thres": ["12500", "11000", "15000"]
                         }

    models_and_feeds = []
    alias_id = {}
    id = 0
    for gpv in general_variations:
        for parameter in neuron_variations:
            for var in neuron_variations[parameter]:
                neuron_params = {k:neuron_variations[k][0] if k != parameter else var for k in neuron_variations}
                models_and_feeds.append(list(construct_network("iter",
                                                          model_file,
                                                          update_neuron_parameters=neuron_params,
                                                          update_general_parameters=gpv
                                                          )
                                        ))
                gpvc = gpv.copy()
                gpvc.update({parameter:var})
                alias_id[id] = gpvc
                models_and_feeds[-1].append(str(id))
                id+=1
    print(alias_id)
    pool = futures.ThreadPoolExecutor(max_workers=os.cpu_count())
    results = pool.map(train, models_and_feeds)
    to_write = []
    for res in list(results):
        save_attention_maps(res['model'], f"experiments_results/{res['log'][0]}")
        to_write.append(res['log'])
    with open("experiments_results/repeat.txt", 'w') as f:
        f.write(str(to_write))
