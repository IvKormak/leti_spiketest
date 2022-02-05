from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Union, Tuple, Dict, List, Type, Any, Callable
import matplotlib.pyplot as plt
import numpy as np
import imageio as iio
import random
import time
import pickle
from utility import *


class Timer:
    __instance = None

    @staticmethod
    def get_instance() -> 'Timer':
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
    activation_function: Callable


class DataFeed:
    def __init__(self, timer: Timer, source=False, deprecate_timer=False, freq=0):
        self.data = ""
        self.timer = timer
        self.freq = freq
        self.deprecate_timer = deprecate_timer
        self.cache = {}

        self.pixels = Defaults.pixels

        if source:
            self.load(source)
        else:
            self.index = 0

    def load(self, source: str):
        self.index = 0
        if source not in self.cache:
            self.diagonal = source in Defaults.diagonals
            file = open(source, 'r')
            self.data = ""
            hexadecimals = "0123456789abcdef"
            while char := file.read(1):
                if char in hexadecimals:
                    self.data += char
            self.cache[source] = self.data
        else:
            self.data = self.cache[source]
        if not self.deprecate_timer:
            self.timer.clock = 0

    def get_pixels(self) -> List[str]:
        return self.pixels

    def __next__(self) -> str:
        try:
            entry = self.data[10 * self.index:10 * (self.index + 1)]
        except IndexError:
            raise StopIteration
        if len(entry) == 0:
            raise StopIteration
        self.index += 1
        if not self.deprecate_timer:
            self.timer.clock = self.parse_aer(entry)[1]
        else:
            if self.diagonal:
                self.timer.clock += np.random.random() * 200 - 100 + (1 / self.freq) * 10 ** 6 * np.sqrt(2)
            else:
                self.timer.clock += np.random.random()*200-100+(1/self.freq)*10**6
        return entry

    def __iter__(self) -> 'DataFeed':
        self.index = 0
        return self

    @staticmethod
    def parse_aer(raw_data: str) -> Tuple[str, int]:
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

    def __init__(self, timer: Timer):
        self.timer = timer
        self.watches = []
        self.logs = {'timestamps': []}

    def add_watch(self, watch: Type[Any], attr: str = 'output'):
        if not hasattr(watch, attr):
            raise AttributeError(f"За объектом нельзя следить, нет свойства {attr}: {watch}")
        self.watches.append((watch, attr))
        self.logs[(watch, attr)] = []

    def remove_watch(self, watch: Type[Any], attr: str = 'output'):
        try:
            for w, a in self.watches:
                if (watch, attr) == (w, a):
                    self.watches.remove((w, a))
        except ValueError:
            raise ValueError(f"За этим объектом не происходит слежения: {watch}")

    def collect_watches(self):
        self.logs['timestamps'].append(self.timer.clock)
        for watch, attr in self.watches:
            self.logs[(watch, attr)].append(getattr(watch, attr))

    def get_logs(self, watch: Type[Any], attr: str = 'output') -> List[Any]:
        if watch == 'timestamps':
            return self.logs['timestamps']
        try:
            return self.logs[(watch, attr)]
        except KeyError:
            raise KeyError(f"За этим объектом не происходит слежения: {(watch, attr)}")

    def reset(self):
        self.logs = {k: [] for k in self.logs}


class Layer(ABC):
    """docstring for Layer"""
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_synapses(self):
        pass

    @abstractmethod
    def update(self, *input_spikes: List[str]):
        pass

    @abstractmethod
    def get_weights(self):
        pass

    @abstractmethod
    def set_random_weights(self):
        pass

    @abstractmethod
    def set_weight(self, weights: Dict[str, int]):
        pass

    @abstractmethod
    def set_param_set(self, param_set: ParameterSet):
        pass

    @abstractmethod
    def mutate(self, *weights: Tuple[Dict[str, int]]):
        pass


class Neuron(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def update(self, *synapses: List[str]):
        pass

    @abstractmethod
    def mutate(self, *weights: Tuple[Dict[str, int]]):
        pass

    @abstractmethod
    def set_random_weights(self):
        pass

    @abstractmethod
    def set_param_set(self, param_set: ParameterSet):
        pass

    @abstractmethod
    def inhibit(self):
        pass


class SpikeNetwork:

    def __init__(self, datafeed: DataFeed, structure: Tuple[Tuple[int]], parameter_set: Dict = {}, learn: bool = False, **layer_args):
        self.layers = []
        self.frame = 0
        if parameter_set != {}:
            parameter_set['activation_function'] = parameter_set['activation_function'](parameter_set['i_thres'])
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
            for layer_num, shape in enumerate(structure):
                layer = STDPLayer(
                    timer=self.timer,
                    shape=shape,
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

    def set_param_set(self, param_set: Dict):
        self.param_set = ParameterSet(**param_set)
        for layer in self.layers:
            layer.set_param_set(self.param_set)

    def reset(self):
        self.frame = 0
        self.timer.clock = 0
        for layer in self.layers:
            layer.reset()

    def set_feed(self, datafeed: DataFeed):
        self.feed = datafeed

    def next(self) -> List[int]:
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
        return self.layers[-1].output_labels

    def get_journals_from_layer(self, layer: int = -1) -> List[List[int]]:
        # выдаем журналы событий с узанного слоя
        # по умолчанию - выходной
        journals = []
        for neuron in self.layers[layer].neurons:
            journals.append(self.logger.get_logs(neuron, 'output_level'))
        return journals

    def calculate_fitness(self):
        """C = []
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
        for neuron_number in neuron_numbers:
            maximum_score_for_trace = 0
            if len(trace_numbers) == 0:
                break
            trace_best_guessed = trace_numbers[0]
            for trace_number in trace_numbers:
                if np.real(F[neuron_number][trace_number]) > maximum_score_for_trace:
                    maximum_score_for_trace = F[neuron_number][trace_number]
                    trace_best_guessed = trace_number
            score += maximum_score_for_trace
            trace_numbers.remove(trace_best_guessed)

        total_ans = np.real(score) + np.imag(score)
        right_ans = np.real(score)
        # +1 в знаменателе решает проблему деления на 0 когда всё очень плохо
        score = (right_ans / (total_ans + 1) - 1 / C.shape[1]) * right_ans
        return score, right_ans, total_ans, f"{right_ans / (total_ans + 1) * 100}%"""

        self.label_neurons()
        self.reset()
        self.logger.reset()
        self.feed.load('resources\\out.bin')
        test(self)


    def label_neurons(self) -> Tuple:
        # определить направления, на которые натренировались нейроны
        neuron_journals = self.get_journals_from_layer()
        teacher_journal = self.logger.get_logs(self.teacher, 'output')
        R = [[] for _ in neuron_journals]
        C = []
        for f, ans in enumerate(teacher_journal):
            if ans:
                for n in range(len(R)):
                    R[n].append(neuron_journals[n][f])
                C.append(ans)
        R, C = np.array(R), np.array(C)
        F = np.matmul(R, C)

        trace_numbers = list(range(C.shape[1]))
        neuron_numbers = list(range(R.shape[0]))
        for neuron_number in neuron_numbers:
            maximum_score_for_trace = 0
            if len(trace_numbers) == 0:
                break
            trace_best_guessed = trace_numbers[0]
            for trace_number in trace_numbers:
                if np.real(F[neuron_number][trace_number]) > maximum_score_for_trace:
                    maximum_score_for_trace = F[neuron_number][trace_number]
                    trace_best_guessed = trace_number
            self.layers[-1].neurons[neuron_number].label = trace_best_guessed

    def get_weights(self) -> List[List[Dict[str, int]]]:
        return [layer.get_weights() for layer in self.layers]

    def set_random_weights(self):
        for layer in self.layers:
            layer.set_random_weights()

    def set_weights(self, weights: List[List[Dict[str, int]]]):
        for l_num, layer in enumerate(self.layers):
            layer.set_weight(weights[l_num])

    def mutate(self, *weights: Tuple[Dict[str, int]]):
        if len(weights) == 0:
            self.set_random_weights()
        else:
            for l_num, layer in enumerate(self.layers):
                # т.к. на входе массив двух геномов, необходимо вырезать из каждого из них
                # кусок соответствующий нужному нам слою
                layer.mutate(*[weight[l_num] for weight in weights])

    def save_attention_maps(self, folder: str):
        attention_maps = []
        pixels_on = Defaults.pixels[1::2]
        pixels_off = Defaults.pixels[::2]
        for w in self.get_weights()[-1]:
            summ = [w[on] if w[on]>w[off] else -1*w[off] for on, off in zip(pixels_on, pixels_off)]

            attention_maps.append(np.array(summ).reshape((28, 28)).transpose())
            #attention_maps.append(np.array([w[k] for k in pixels_off]).reshape((28,28)).transpose())
            #attention_maps.append(np.array([w[k] for k in pixels_on]).reshape((28,28)).transpose())

        for i, map in enumerate(attention_maps):
            plt.close()
            plt.imshow(map)
            plt.savefig(f"{folder}/att_map_{i}_{Defaults.traces[self.layers[-1].neurons[i].label]}.png")


class STDPLayer(Layer):
    def __init__(self, timer, shape: Tuple[int], layer_number: int, wta:bool = False, decimation_coefficient:int = 0, inhib_radius:int = 1, **neuron_args):
        self.timer = timer
        self.shape = shape
        self.layer_number = layer_number

        self.inhib_radius = inhib_radius
        self.decimation_coefficient = decimation_coefficient

        self.output = []
        self.output_labels = []
        self.neurons = np.array([STDPNeuron(timer, **neuron_args) for __ in range(self.shape[1]*self.shape[0])])

        self.wta = wta

    def reset(self):
        for neuron in self.neurons:
            neuron.reset()

    def mtx_neuron(self, row, col):
        if row < 0 or col < 0 or row > self.shape[1] or col > self.shape[0]:
            raise IndexError("Выход за рамки матрицы нейронов")
        return self.neurons[row*self.shape[1]+col]

    def mtx_neighbpurhood(self, radius, row, col):
        return [(row+i,col+j) for i in range(-radius, radius+1) for j in range(-radius+abs(i), radius-abs(i)+1)]

    def update(self, *input_spikes):
        if self.decimation_coefficient:
            active_neurons = [n for n in self.neurons if random.random() > self.decimation_coefficient]
        else:
            active_neurons = self.neurons
        wta = self.wta
        output = np.array([[active_neurons[x*self.shape[1]+y].update(*input_spikes)
                            for y in range(self.shape[1])]
                           for x in range(self.shape[0])])
        input = np.array([[active_neurons[x*self.shape[1]+y].input_level
                            for y in range(self.shape[1])]
                           for x in range(self.shape[0])])
        self.output = output.flatten()
        self.output_labels = np.array([n.label for n in active_neurons if n.output_level])

        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if output[row][col]:
                    for i, j in self.mtx_neighbpurhood(self.inhib_radius, row, col):
                        try:
                            self.mtx_neuron(row+i, col+j).inhibit()
                        except IndexError:
                            pass

    def get_synapses(self):
        return [
            format(self.layer_number, '02x') +
            format(n, '02x')
            for n in range(len(self.neurons))
        ]

    def get_weights(self):
        return [neuron.weights for neuron in self.neurons] #возвращает веса, смещённые на константу
        #чтобы получить реальные значения, необходимо вычесть из них количество срабатываний нейрона

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


class STDPNeuron(Neuron):
    """Implementaion for a spike NN neuron"""

    def __init__(self,
                 timer,
                 parameter_set,
                 synapses={},
                 learn=False):
        """инициализация начальных параметров"""
        self.timer = timer
        self.input_level = 0
        self.output_level = 0
        self.param_set = parameter_set
        self.weights = {k: 800 for k in synapses}
        self.synapses = np.array(synapses)
        self.learn = learn

        self.i_thres = parameter_set.i_thres
        self.refractory = False
        self.ltp_times = {}
        self.t_spike = -1
        self.t_last_spike = -1
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1

        self.times_fired = 0

        self.label = 0

    def reset(self):
        self.input_level = 0
        self.output_level = 0
        self.t_spike = -1
        self.t_last_spike = -1
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1
        self.ltp_times = {}


    def update(self, *synapses):
        """обработать пришедшие данные и обновить состояние нейрона. Основная функция"""
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.timer.clock
        if self.timer.clock <= self.inhibited_by:
            return 0
        self.output_level = 0
        self.refractory = False
        for synapse in synapses[0]:
            self.t_last_spike = self.t_spike
            self.t_spike = self.timer.clock
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.param_set.t_leak)
            self.input_level += self.weights[synapse]
            self.input_level -= self.times_fired*self.param_set.a_dec
            self.ltp_times[synapse] = self.timer.clock + self.param_set.t_ltp
        self.output_level = self.input_level > self.i_thres
        if self.output_level:
            self.times_fired += 1
            self.input_level = 0
            self.refractory = True
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn:
                not_rotten = [k for k in self.ltp_times.keys() if self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc + self.param_set.a_dec
                self.ltp_times = {}
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




def init_model(feed_options = {}, model_options = {}, parameter_set=False):
    feed_opts = {'source': 'resources\\out.bin',
                 'timer': Timer.get_instance()}

    feed_opts.update(feed_options)
    feed = DataFeed(**feed_opts)

    if not parameter_set:
        parameter_set = Defaults.param_set

    model_opts = {'datafeed': feed,
                  'parameter_set': parameter_set
                  }
    model_opts.update(model_options)

    model = SpikeNetwork(**model_opts)
    print(model.param_set)

    return model

def test(model):
    arrows = pickle.load(open('resources\\arrows.bin', 'rb'))

    white_square = np.multiply(np.ones((28,28)), 255)
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
        data = model.raw_data
        synapse = DataFeed.parse_aer(data)[0][0]
        x_coord = int(synapse[0:2], base=16)
        y_coord = int(synapse[2:4], base=16)
        color = (synapse[4] == '0') * 255
        current_frame[y_coord][x_coord] = color
        frames.append(np.concatenate((current_frame, current_arrow)).astype(np.uint8))
        frames_shown += 1
        frame = model.next()

    iio.mimwrite('animation2.gif', frames, fps=60)