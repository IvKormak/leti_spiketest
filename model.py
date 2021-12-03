from abc import ABC, abstractmethod
from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle
from utility import *


class Timer:
    __instance = None

    @staticmethod
    def get_instance():
        """ Static access method. """
        if Timer.__instance is None:
            Timer()
        return Timer.__instance

    def __init__(self):
        """ Virtually private constructor. """
        self.clock = 0
        if Timer.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            Timer.__instance = self


@dataclass
class ParameterSet:
    i_thres: int
    t_ltp: int
    t_refrac: int
    t_inhibit: int
    t_leak: int
    w_min: int
    w_max: int
    a_dec: int
    a_inc: int
    randmut: int


class DataFeed:
    def __init__(self, timer, source=False):
        self.data = ""
        self.timer = timer
        self.cache = {}

        self.pixels = Defaults.pixels

        if source:
            self.load(source)
        else:
            self.index = 0

    def load(self, source: str):
        self.index = 0
        if source not in self.cache:
            file = open(source, 'r')
            self.data = ""
            hexadecimals = "0123456789abcdef"
            while char := file.read(1):
                if char in hexadecimals:
                    self.data += char
            self.cache[source] = self.data
        else:
            self.data = self.cache[source]
        self.timer.clock = 0

    def get_pixels(self):
        return self.pixels

    def __next__(self):
        try:
            entry = self.data[10 * self.index:10 * (self.index + 1)]
        except IndexError:
            raise StopIteration
        if len(entry) == 0:
            raise StopIteration
        self.index += 1
        self.timer.clock = self.parse_aer(entry)[1]
        return entry

    def __iter__(self):
        self.index = 0
        return self

    @staticmethod
    def parse_aer(raw_data):
        data = []
        time = 0
        if isinstance(raw_data, str):
            raw_data = int(raw_data, base=16)
        if isinstance(raw_data, int):
            raw_data = [raw_data]
        for entry in raw_data:
            synapse = entry >> Defaults.time_bits
            synapse = synapse << 3
            synapse = format(synapse, '05x')
            time = entry & Defaults.time_mask
            # print(format(raw_data, "#040b"))
            # print(synapse,
            #    raw_data&defaults.time_mask)
            data.append(synapse)
        return data, time


class Teacher:
    def __init__(self):
        self.output = None


class Logger:
    """Класс логгера"""

    def __init__(self, timer):
        self.timer = timer
        self.watches = []
        self.logs = {'timestamps': []}

    def add_watch(self, watch, attr='output'):
        if not hasattr(watch, attr):
            raise AttributeError(f"За объектом нельзя следить, нет свойства {attr}: {watch}")
        self.watches.append((watch, attr))
        self.logs[watch] = []

    def remove_watch(self, watch):
        try:
            for w, a in self.watches:
                if watch == w:
                    self.watches.remove((w, a))
        except ValueError:
            raise ValueError(f"За этим объектом не происходит слежения: {watch}")

    def collect_watches(self):
        self.logs['timestamps'].append(self.timer.clock)
        for watch, attr in self.watches:
            self.logs[watch].append(getattr(watch, attr))

    def get_logs(self, watch):
        try:
            return self.logs[watch]
        except KeyError:
            raise KeyError(f"За этим объектом не происходит слежения: {watch}")

    def reset(self):
        self.logs = {k: [] for k in self.logs}


class SpikeNetwork:

    def __init__(self, datafeed, structure, parameter_set=False, learn=False, **layer_args):
        self.layers = []
        self.frame = 0
        if parameter_set:
            self.param_set = ParameterSet(**parameter_set)
        else:
            self.param_set = ParameterSet(**Defaults.param_set)

        self.feed = datafeed
        self.raw_data = ""
        self.timer = datafeed.timer

        self.logger = Logger(self.timer)

        if learn:
            self.teacher = Teacher()
            self.logger.add_watch(self.teacher)

        try:
            synapses = self.feed.get_pixels()
            for layer_num, neuron_num in enumerate(structure):
                layer = STDPLayer(
                    timer=self.timer,
                    neuron_count=neuron_num,
                    layer_number=layer_num + 1,
                    synapses=synapses,
                    parameter_set=self.param_set,
                    learn=learn
                )
                synapses = layer.get_synapses()
                for neuron in layer.neurons:
                    self.logger.add_watch(neuron, 'output_level')
                self.layers.append(layer)
        except KeyError:
            raise Exception("Невозможно инициализировать сеть не указав структуру!")

    def set_param_set(self, param_set):
        self.param_set = ParameterSet(**param_set)
        for layer in self.layers:
            layer.set_param_set(self.param_set)

    def reset(self):
        super().reset()
        self.frame = 0

    def set_feed(self, datafeed: DataFeed):
        self.feed = datafeed

    def __iter__(self):
        return self

    def next(self):
        # читаем следующий сигнал с камеры
        try:
            self.raw_data = next(self.feed)
        except StopIteration:
            return False
        # парсим данные
        input_data, t = self.feed.parse_aer(self.raw_data)  # cutting bits 0:22
        input_data = [input_data, ]
        # указываем слоям срабатывать по очереди и передаём результаты вглубь
        for layer in self.layers:
            layer.update(*input_data)
            input_data = layer.output
        # увеличиваем счётчик шагов, выдаём значение с выходного слоя
        self.frame += 1
        self.logger.collect_watches()
        return input_data

    def get_journals_from_layer(self, layer=-1):
        # выдаем журналы событий с узанного слоя
        # по умолчанию - выходной
        journals = []
        for neuron in self.layers[layer].neurons:
            journals.append(self.logger.get_logs(neuron))
        return journals

    def calculate_fitness(self):
        # считаем фитнесс функцию от числа правильных и неправильных ответов
        neuron_journals = self.get_journals_from_layer()
        teacher_journal = self.logger.get_logs(self.teacher)
        R = [[] for _ in neuron_journals]
        C = []
        for f, ans in enumerate(teacher_journal):
            if ans:
                for n in range(len(R)):
                    R[n].append(neuron_journals[n][f])
                C.append(ans)
        R, C = np.array(R), np.array(C)
        F = np.matmul(R, C)
        score = 0
        trace_numbers = list(range(C.shape[1]))
        neuron_numbers = list(range(R.shape[0]))
        random.shuffle(neuron_numbers)
        preferred_order = [0 for _ in neuron_numbers]
        for neuron_number in neuron_numbers:
            maximum_score_for_trace = 0
            trace_best_guessed = trace_numbers[0]
            for trace_number in trace_numbers:
                if np.real(F[neuron_number][trace_number]) > maximum_score_for_trace:
                    maximum_score_for_trace = F[neuron_number][trace_number]
                    trace_best_guessed = trace_number
            score += maximum_score_for_trace
            trace_numbers.remove(trace_best_guessed)
            preferred_order[neuron_number] = trace_best_guessed
        # установим выходные нейроны в таком порядке чтобы
        # порядковый номер нейрона соответствовал порядковому
        # номеру направления, которое он определяет лучше всего
        self.layers[-1].rearrange_neurons(preferred_order)
        total_ans = np.real(score) + np.imag(score)
        right_ans = np.real(score)
        # +1 в знаменателе решает проблему деления на 0 когда всё очень плохо
        score = (right_ans / (total_ans + 1) - 1 / C.shape[1]) * right_ans
        return score, right_ans, total_ans, f"{right_ans / (total_ans + 1) * 100}%"

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_random_weights(self):
        for layer in self.layers:
            layer.set_random_weights()

    def set_weights(self, weights: list):
        for l_num, layer in enumerate(self.layers):
            layer.set_weight(weights[l_num])

    def mutate(self, *weights):
        if len(weights) == 0:
            self.set_random_weights()
        else:
            for l_num, layer in enumerate(self.layers):
                # т.к. на входе массив двух геномов, необходимо вырезать из каждого из них
                # кусок соответствующий нужному нам слою
                layer.mutate(*[weight[l_num] for weight in weights])

    def save_attention_maps(self, folder):
        attention_maps = []
        pixels_on = Defaults.pixels[1::2]
        pixels_off = Defaults.pixels[::2]
        for w in self.get_weights()[-1]:
            attention_maps.append(np.array([w[k] for k in pixels_off]).reshape((28,28)).transpose())
            attention_maps.append(np.array([w[k] for k in pixels_on]).reshape((28,28)).transpose())

        for i, map in enumerate(attention_maps):
            plt.close()
            plt.imshow(map)
            plt.savefig(f"{folder}/att_map_{Defaults.traces[i//2]}{i%2}.png")


class Layer(ABC):
    """docstring for Layer"""

    @abstractmethod
    def get_synapses(self):
        pass

    @abstractmethod
    def update(self, *input_spikes):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_random_weights(self):
        pass

    @abstractmethod
    def set_weight(self, weights):
        pass

    @abstractmethod
    def set_param_set(self, param_set):
        pass

    @abstractmethod
    def mutate(self, *weights):
        pass

    @abstractmethod
    def rearrange_neurons(self, new_order):
        pass


class Neuron(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *synapses):
        pass

    @abstractmethod
    def mutate(self, *weights: list):
        pass

    @abstractmethod
    def set_random_weights(self):
        pass

    @abstractmethod
    def set_param_set(self, param_set):
        pass

    @abstractmethod
    def inhibit(self):
        pass


class STDPLayer(Layer):
    def __init__(self, timer, neuron_count, layer_number, wta=False, **neuron_args):
        self.timer = timer
        self.neuron_count = neuron_count
        self.layer_number = layer_number

        self.output = []
        self.neurons = [STDPNeuron(timer, **neuron_args) for _ in range(self.neuron_count)]
        self.wta = wta

    def update(self, *input_spikes):
        neurons = self.neurons
        wta = self.wta
        output = [neuron.update(*input_spikes) for neuron in neurons]

        self.output = output

        for output, fired_neuron in zip(output, neurons):
            if output:
                for neuron in neurons:
                    if neuron != fired_neuron:
                        neuron.inhibit()
                        if wta:
                            neuron.input_level = 0

    def get_synapses(self):
        return [
            format(self.layer_number, '02x') +
            format(n, '02x')
            for n in range(len(self.neurons))
        ]

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons]

    def set_random_weights(self):
        for neuron in self.neurons:
            neuron.set_random_weights()

    def set_weight(self, weights):
        for neuron, weight in zip(self.neurons, weights):
            neuron.weights = weight

    def set_param_set(self, param_set):
        for neuron in self.neurons:
            neuron.set_param_set(param_set)

    def mutate(self, *weights):
        for n_num, neuron in enumerate(self.neurons):
            neuron.mutate(*[weight[n_num] for weight in weights])

    def rearrange_neurons(self, new_order):
        new_neurons = [0] * len(new_order)
        for neuron_num, trace_number in enumerate(new_order):
            new_neurons[trace_number] = self.neurons[neuron_num]
        self.neurons = new_neurons


class STDPNeuron(Neuron):
    """Implementaion for a spike NN neuron"""

    def __init__(self,
                 timer,
                 parameter_set,
                 synapses={},
                 learn=False,
                 activation_function=False):
        """инициализация начальных параметров"""
        self.timer = timer
        self.input_level = 0
        self.output_level = 0
        self.param_set = parameter_set
        self.weights = {k: 800 for k in synapses}
        self.learn = learn

        if activation_function:
            self.activation_function = activation_function
        else:
            self.activation_function = ActivationFunctions.DeltaFunction(self.param_set.i_thres)

        self.refractory = False
        self.ltp_synapses = {}  # записано, до какого момента при срабатывании синапс увеличит вес
        self.t_spike = 0
        self.t_last_spike = 0
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1

    def reset(self):
        self.input_level = 0
        self.output_level = 0
        self.t_spike = -1
        self.t_last_spike = -1
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1
        self.ltp_synapses = {}

    def update(self, *synapses):
        """обработать пришедшие данные и обновить состояние нейрона. Основная функция"""
        super().update(*synapses)
        if self.timer.clock <= self.inhibited_by:
            return 0
        self.output_level = 0
        self.refractory = False
        for synapse in synapses[0]:
            self.t_last_spike, self.t_spike = self.t_spike, self.timer.clock
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.param_set.t_leak)
            self.input_level += self.weights[synapse]
            self.ltp_synapses[synapse] = self.timer.clock + self.param_set.t_ltp
        self.output_level = self.activation_function(self.input_level)
        if self.output_level:
            self.input_level = 0
            self.refractory = True
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                for synapse, time in self.ltp_synapses.items():
                    if time >= self.t_spike - self.param_set.t_ltp:
                        self._synapse_inc_(synapse)
                    else:
                        self._synapse_dec_(synapse)
                self.ltp_synapses = {}
        return self.output_level

    def mutate(self, *weights: list):
        for synapse in self.weights:
            if random.random() > self.param_set.randmut:
                n = round((len(weights) - 1) * random.random())
                self.weights[synapse] = weights[n][synapse]
            else:
                self.weights[synapse] = random.randrange(self.param_set.w_min, self.param_set.w_max)
            if self.weights[synapse] > self.param_set.w_max:
                self.weights[synapse] = self.param_set.w_max
            if self.weights[synapse] < self.param_set.w_min:
                self.weights[synapse] = self.param_set.w_min

    def _synapse_inc_(self, synapse: int):
        """усилить связь с синапсами, сработавшими прямо перед срабатыванием нейрона"""
        self.weights[synapse] += self.param_set.a_inc
        if self.weights[synapse] > self.param_set.w_max:
            self.weights[synapse] = self.param_set.w_max

    def _synapse_dec_(self, synapse: int):
        """ослабить связи с синапсами, не сработавшими перед срабатыванием нейрона"""
        self.weights[synapse] -= self.param_set.a_dec
        if self.weights[synapse] < self.param_set.w_min:
            self.weights[synapse] = self.param_set.w_min

    def inhibit(self):
        # эта функция вызывается только из модели в случае срабатывания другого нейрона
        # так как нейрон может быть неактивен, необходимо проверить сроки
        if (self.inhibited_on < self.timer.clock) or not self.refractory:  # не надо дважды ингибировать в один фрейм
            self.inhibited_on = self.timer.clock
            self.inhibited_by = self.timer.clock + self.param_set.t_inhibit
        elif self.refractory and self.inhibited_by - self.param_set.t_refrac < self.inhibited_on:
            # в случае когда нейрон уже находится
            # в состоянии восстановления увеличиваем срок восстановления
            # но проверяем, не добавили ли к этому нейрону уже время ингибиции.
            # inhibited_by никогда не может превышать clock больше чем на сумму t_inhibit и t_refrac
            self.inhibited_by += self.param_set.t_inhibit

    def set_random_weights(self):
        self.weights = {k: random.randrange(self.param_set.w_min, self.param_set.w_max) for k in self.weights}

    def set_param_set(self, param_set):
        self.param_set = param_set


def init_model(feed_options = {}, model_options = {}, parameters_set=False):
    feed_opts = {'source': 'resources\\out.bin',
                 'timer': Timer.get_instance()}

    feed_opts.update(feed_options)
    feed = DataFeed(**feed_opts)

    if not parameters_set:
        parameters_set = Defaults.param_set

    model_opts = {'datafeed': feed,
                  'parameter_set': parameters_set
                  }
    model_opts.update(model_options)

    model = SpikeNetwork(**model_opts)

    return model
