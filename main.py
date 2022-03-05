from dataclasses import dataclass, field
from collections import namedtuple
import matplotlib.pyplot as plt
import numpy as np
import configparser as cm
import AERGen as ag
import random
import os
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

    def next_events(self):
        events = []
        if self.feed_type == "iter":
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

LayerParametersSet = namedtuple("LayerParametersSet", ["inhibit_radius",
                                                       "epoch_length",
                                                       "execution_thres",
                                                       "terminate_on_epoch"])


@dataclass
class Model:
    neuron_parameters_set: NeuronParametersSet
    layer_parameters_set: LayerParametersSet
    state: dict = field(default_factory=dict)
    outputs: list = field(default_factory=list)
    time: int = 0
    logs: list = field(default_factory=list)
    layers: list = field(default_factory=list)


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

        self.age = 0

    def copy(self, model):
        return Neuron(model, output_address=self.output_address, inputs=self.inputs, learn=self.learn)

    def update(self):
        self.output_level = 0
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.model.time
        if self.model.time <= self.inhibited_by:
            return 0

        state = self.model.state #копируем для ускорения доступа
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
            self.output_level = int(self.input_level>self.param_set.i_thres)
        if self.output_level:

            self.times_fired += 1

            min_level = self.param_set.w_min
            max_level = self.param_set.w_max
            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                rest = [k for k in self.inputs if k not in not_rotten]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc
                    if self.weights[synapse] > max_level:
                        self.weights[synapse] = max_level
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
        self.age = 0

    def attention_map(self):
        pixels_on = Defaults.pixels[1::2]
        pixels_off = Defaults.pixels[::2]
        all_pixels = [self.weights[on] if self.weights[on] > self.weights[off]
                      else -1 * self.weights[off]
                      for on, off in zip(pixels_on, pixels_off)]
        return self.output_address, self.label, np.array(all_pixels).reshape((28, 28)).transpose()

    def set_weights(self, weights):
        self.weights = weights.copy()

    def random_weights(self):
        return {i: random.random() * self.param_set.w_random * (self.param_set.w_max - self.param_set.w_min) for i in self.inputs}


def construct_network(feed_type, source_file="network1.txt", learn=True, update_neuron_parameters={}, update_layer_parameters={}, pool_size=10):
    config = cm.ConfigParser()
    with open(source_file, 'r') as f:
        config.read_file(f)
    config["NEURON PARAMETERS"].update(update_neuron_parameters)
    config["LAYER PARAMETERS"].update(update_layer_parameters)
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
    lps = LayerParametersSet(int(config["LAYER PARAMETERS"]["inhibit_radius"]),
                             int(config["LAYER PARAMETERS"]["epoch_length"]),
                             int(config["LAYER PARAMETERS"]["execution_thres"]),
                             int(config["LAYER PARAMETERS"]["terminate_on_epoch"]))
    model = Model(neuron_parameters_set=nps, layer_parameters_set=lps)
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
    training_pool = []
    for _ in range(pool_size):
        t_model = Model(state=model.state.copy(),
                        outputs=model.outputs.copy(),
                        neuron_parameters_set=model.neuron_parameters_set,
                        layer_parameters_set=model.layer_parameters_set,
                        )
        t_model.layers = [{'neurons': [n.copy(t_model) for n in l['neurons']], 'shape': l['shape']} for l in model.layers]
        training_pool.append(t_model)
    return model, training_pool, DataFeed(feed_type, model)


def load_network(model, feed_type, learn=False):
    model.time = 0
    model.state = {s: 0 for s in model.state}
    model.logs = []
    for layer in model.layers:
        for neuron in layer['neurons']:
            neuron.learn = learn
    return DataFeed(feed_type, model)


def next_training_cycle(training_pool, feed):
    next_ev = feed.next_events()
    for donor_model in training_pool:
        if feed.terminate:
            for layer in donor_model.layers:
                for neuron in layer['neurons']:
                    neuron.age += 1
        else:
            donor_model.time = next_ev[0].time
            for ev in next_ev:
                donor_model.state[ev.address] = 1

            for layer in donor_model.layers:
                layer_update(donor_model, layer)

            for synapse in donor_model.outputs:
                if donor_model.state[synapse]:
                    donor_model.logs.append((synapse, feed.source))

            donor_model.state = {k: 0 for k in donor_model.state.keys()}
    return not feed.terminate


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


def label_neurons(donor_model):
    neuron_journals = [l[0] for l in donor_model.logs[len(donor_model.logs)*8//10:]]
    teacher_journals = [l[1] for l in donor_model.logs[len(donor_model.logs)*8//10:]]
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
    s_r = 0
    s_i = 0
    for f in F.flatten():
        if np.real(f) > 0:
            s_r += np.real(f)
            s_i += np.imag(f)
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

    countlabels = {}

    for neuron in donor_model.layers[-1]['neurons']:
        if neuron.output_address in rename_dict:
            neuron.label = rename_dict[neuron.output_address]
        if neuron.label:
            if neuron.age >= donor_model.layer_parameters_set.epoch_length:
                neuron.learn = False
            if not neuron.label in countlabels:
                countlabels[neuron.label] = 0
            countlabels[neuron.label] += 1

    donor_model.logs = []

    return countlabels


def select_weights_from_pool(layers_pool, dup_thres):
    learnt_categories = {}
    neuron_pool = np.array(layers_pool).flatten()
    for n in neuron_pool:
        if n.label:
            if n.label not in learnt_categories:
                learnt_categories[n.label] = {}
                learnt_categories[n.label]['weights'] = [n.weights,]
                learnt_categories[n.label]['count'] = 1
            if n.label in learnt_categories:
                if learnt_categories[n.label]['count'] < dup_thres:
                    learnt_categories[n.label]['weights'].append(n.weights)
                    learnt_categories[n.label]['count'] += 1
    return learnt_categories


def fill_model_from_pool(model: Model, training_pool: [Model]):
    filled_positions = []
    for i, layer in enumerate(model.layers):
        layers_pool = [m.layers[i]['neurons'] for m in training_pool]
        learnt = select_weights_from_pool(layers_pool, model.layer_parameters_set.execution_thres)
        print(learnt.keys())
        already_there = set([neuron.label for neuron in layer['neurons'] if neuron.label])
        already_there_count = {k: [n.label for n in layer['neurons']].count(k) for k in already_there}
        print(already_there_count)

        for k in already_there_count.keys():
            if k in learnt:
                learnt[k]['count'] -= already_there_count[k]

        select = []
        free_neurons = [neuron for neuron in layer['neurons'] if not neuron.label]

        for label, e in learnt.items():
            if e['count'] > 0:
                for i in range(e['count']):
                    select.append((e['weights'][i], label))

        for recipient, (graft, label) in zip(free_neurons, select):
            recipient.set_weights(graft)
            recipient.label = label

        already_there = set([neuron.label for neuron in layer['neurons'] if neuron.label])
        already_there_count = {k: [n.label for n in layer['neurons']].count(k) for k in already_there}
        print(already_there_count)

    return already_there_count


def delete_duplicate_neurons(model, countlabels):
    for neuron in model.layers[-1]['neurons']:
        if neuron.label:
            if countlabels[neuron.label] > model.layer_parameters_set.execution_thres:
                countlabels[neuron.label] -= 1
                neuron.reset()
                neuron.learn = True


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


def show_attention_maps(model, folder: str):
    attention_maps = []
    if not os.path.exists(folder):
        os.mkdir(folder)
    for n in model.layers[-1]['neurons']:
        attention_maps.append(n.attention_map())

    for address, label, map in attention_maps:
        plt.close()
        plt.imshow(map)
        plt.show()


def reset(model, feed):
    model.time = 0
    feed.time_offset = 0
    model.state = {s: 0 for s in model.state}
    model.logs = []
    for layer in model.layers:
        for neuron in layer['neurons']:
            neuron.reset()


if __name__ == "__main__":

    sets1000 = ["traces/b-t-1000.bin", "traces/l-r-1000.bin", "traces/r-l-1000.bin", "traces/t-b-1000.bin"]
    sets3000 = ["traces/b-t-3000.bin", "traces/l-r-3000.bin", "traces/r-l-3000.bin", "traces/t-b-3000.bin"]
    sets5000 = ["traces/b-t-5000.bin", "traces/l-r-5000.bin", "traces/r-l-5000.bin", "traces/t-b-5000.bin"]
    sets100 = ["traces/b-t-100.bin", "traces/l-r-100.bin", "traces/r-l-100.bin", "traces/t-b-100.bin"]
    sets500 = ["traces/b-t-500.bin", "traces/l-r-500.bin", "traces/r-l-500.bin", "traces/t-b-500.bin",
               "traces/bl-tr-500.bin", "traces/tl-br-500.bin", "traces/br-tl-500.bin", "traces/tr-bl-500.bin"]

    sweep = [("resources/models/network2_c.txt", "fukk")]*1

    for file, folder in sweep:
        target_model, training_pool, feed = construct_network("iter", file,
                                                              update_neuron_parameters={'epoch_length': '50'},
                                                              pool_size=5)
        chosenSets = sets500
        labels = {}
        datasets = []
        epoch_count = 0
        learned_fully = False

        for path in chosenSets:
            with open(path, 'r') as f:
                datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))

        reset(target_model, feed)

        while not learned_fully or epoch_count < target_model.layer_parameters_set.terminate_on_epoch:
            [reset(donor_model, feed) for donor_model in training_pool]
            epoch_count += 1
            for path in chosenSets:
                with open(path, 'r') as f:
                    datasets.append((path, [ag.aer_decode(ev) for ev in f.readline().split(' ')]))

            for n in range(target_model.layer_parameters_set.epoch_length):
                alias, dataset = random.choice(datasets)
                feed.load(alias, dataset)
                while next_training_cycle(training_pool, feed):
                    pass

            for donor_model in training_pool:
                label_neurons(donor_model)

            labels = fill_model_from_pool(target_model, training_pool)
            print(labels)

            learned_fully = len(labels) == len(chosenSets)

        print(f"распознано: {len(labels)}, {labels}")

        save_attention_maps(target_model, f"experiments_results/{folder}")