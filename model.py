from camera_feed import *

import matplotlib.pyplot as plt
import numpy as np
import random
import time
import pickle

class Timer(object):
    """docstring for Timer"""

    def __init__(self, time=0):
        super(Timer, self).__init__()
        self.clock = time
        self.start_time = time
        self.listeners = []

    def set_time(self, time):
        self.clock = time
        self.announce()

    def get_time(self):
        return self.clock

    def reset(self):
        self.set_time(0)
        for listener in self.listeners:
            listener.reset()

    def hard_reset(self):
        self.set_time(0)
        for listener in self.listeners:
            listener.hard_reset()

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listener):
        self.listeners.remove(listener)

    def announce(self):
        for listener in self.listeners:
            listener.update_clock(self.clock)


class SyncedObject(object):
    def __init__(self, *args, **kwargs):
        self.clock = 0
        try:
            kwargs['timer'].add_listener(self)
        except KeyError:
            raise Exception("Невозможно инициализировать объект без таймера!")

    def update_clock(self, time:int):
        self.clock = time

    def reset(self):
        self.clock = 0

    def hard_reset(self):
        self.reset()

class Logger(SyncedObject):
    """Класс логгера. Он должен инициализироваться до сети!"""
    def __init__(self, *args, **kwargs):
        super(Logger, self).__init__(*args, **kwargs)
        self.watches = []
        self.logs = {}

    def add_watch(self, watch):
        if not hasattr(watch, 'output'):
            raise AttributeError("За объектом нельзя следить, нет свойства output")
        self.watches.append(watch)
        self.logs[watch] = {}

    def collect_watches(self):
        clock = self.clock
        for watch in self.watches:
            self.logs[watch][clock] = watch.output

    def get_logs(self, watch):
        try:
            return self.logs[watch]
        except KeyError:
            raise KeyError("За этим объектом не происходит слежения")

    def update_clock(self, time:int):
        self.collect_watches()
        super(Logger, self).update_clock(time)

class Teacher:
    def __init__(self):
        self.output = 0

class SpikeNetwork(SyncedObject):
    """docstring for Network"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layers = []
        self.frame = 0

        try:
            self.feed = kwargs['datafeed']
        except KeyError:
            raise Exception("Невозможно инициализировать сеть без потока входных данных!")

        try:
            self.logger = kwargs['logger']
        except KeyError:
            raise Exception("Невозможно инициализировать сеть без логгера!")

        if 'learn' in kwargs:
            if kwargs['learn']:
                try:
                    self.teacher = kwargs['teacher']
                except KeyError:
                    raise Exception("Невозможно инициализировать сеть без учителя!")

        try:
            synapses = self.feed.get_pixels()
            for layer_num, neuron_num in enumerate(kwargs['structure']):
                weights = {x: 0 for x in synapses}
                layer = STDPLayer(
                        neuron_number=neuron_num,
                        layer_number=layer_num + 1,
                        weights=weights,
                        **kwargs
                )
                synapses = layer.get_synapses()
                self.layers.append(layer)
        except KeyError:
            raise Exception("Невозможно инициализировать сеть не указав структуру!")

    def set_param_set(self, param_set):
        for layer in self.layers:
            layer.set_param_set(param_set)

    def reset(self):
        super().reset()
        self.frame = 0

    def cycle_through(self):
        # впустую прогоняются все данные для обучения
        for o in self:
            pass

    def set_feed(self, datafeed: DataFeed):
        self.feed = datafeed

    def __next__(self):
        # читаем следующий сигнал с камеры
        self.raw_data = next(self.feed)
        # парсим данные
        input_data, t = parse_aer(self.raw_data)  # cutting bits 0:22
        # указываем слоям срабатывать по очереди и передаём результаты вглубь
        for layer in self.layers:
            layer.update(input_data)
            input_data = layer.output
        # увеличиваем счётчик шагов, выдаём значение с выходного слоя
        self.frame += 1
        return input_data

    def __iter__(self):
        # для соответствия стандарту итератора
        return self

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
        for key in teacher_journal.keys():
            if key > 0:
                for n in range(len(R)):
                    R[n].append(neuron_journals[n][key])
                C.append(teacher_journal[key])
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
        total_ans = np.real(score)+np.imag(score)
        right_ans = np.real(score)
        # +1 в знаменателе решает проблему деления на 0 когда всё очень плохо
        score = (right_ans/(total_ans+1)-1/C.shape[1])*right_ans
        return (score, right_ans, total_ans, f"{right_ans/(total_ans+1)*100}%")

    def get_weights(self):
        return [layer.get_weights() for layer in self.layers]

    def set_random_weights(self):
        for layer in self.layers:
            layer.set_random_weights()

    def set_weights(self, weight:list):
        for l_num, layer in enumerate(self.layers):
            layer.set_weight(weight[l_num])

    def mutate(self, *weights):
        if len(weights) == 0:
            self.set_random_weights()
        else:
            for l_num, layer in enumerate(self.layers):
                # т.к. на входе массив двух геномов, необходимо вырезать из каждого из них
                # кусок соответствующий нужному нам слою
                layer.mutate(*[weight[l_num] for weight in weights])

    def stop_learning(self):
        for layer in self.layers:
            layer.stop_learning()

    def start_learning(self):
        for layer in self.layers:
            layer.start_learning()

    def save_attention_maps(self, folder):
        attention_maps = [np.array(list(x.values())).reshape(28,56) for x in self.get_weights()[-1]]
        for i, map in enumerate(attention_maps):
            plt.imshow(map)
            plt.savefig(f"{folder}/att_map_{Defaults.traces[i]}.png")


class Layer(SyncedObject):
    """docstring for Layer"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.neuron_number = kwargs['neuron_number']
        except KeyError:
            raise Exception("Необходимо указать число нейронов в параметре neuron_number!")
        self.neurons = []
        try:
            self.layer_number = kwargs['layer_number']
        except KeyError:
            raise Exception("Необходимо указать номер слоя в параметре layer_number!")

        if 'wta' in kwargs:
            self.wta = kwargs['wta']

        self.output = []

    def reset(self):
        super().reset()

    def hard_reset(self):
        super().hard_reset()

    def get_synapses(self):
        return [
            format(self.layer_number, '02x') +
            format(n, '02x')
            for n in range(len(self.neurons))
        ]

    def update(self, input_spikes):
        neurons = self.neurons

        if isinstance(input_spikes, int):
            input_spikes = list([input_spikes])
        output = [neuron.update(input_spikes) for neuron in random.sample(neurons, len(neurons))]

        self.output = output

    def get_input_journals(self):
        return [neuron.get_input_journal() for neuron in self.neurons]

    def get_output_journals(self):
        return [neuron.get_output_journal() for neuron in self.neurons]

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

    def stop_learning(self):
        for neuron in self.neurons:
            neuron.learn = False

    def start_learning(self):
        for neuron in self.neurons:
            neuron.learn = True


class STDPLayer(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            self.neurons = [STDPNeuron(*args, **kwargs) for _ in range(self.neuron_number)]
        except:
            raise Exception("Недостаточно параметров для инициализации нейронов!")
        if 'wta' in kwargs:
            self.wta = kwargs['wta']
        else:
            self.wta = False

    def update(self, input_spikes):
        super().update(input_spikes)
        output = self.output
        neurons = self.neurons
        wta = self.wta

        for output, fired_neuron in zip(output, neurons):
            if output:
                for neuron in neurons:
                    if neuron != fired_neuron:
                        neuron.inhibit()
                        if wta:
                            neuron.reset_input()


IS_FIRED = True
NOT_FIRED = False


class Neuron(SyncedObject):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.input_level = 0
        self.output = 0

        if 'weights' in kwargs:
            self.weights = kwargs['weights']
        else:
            self.weights = {}

        if 'learn' in kwargs:
            self.learn = kwargs['learn']
        else:
            self.learn = False

        try:
            kwargs['logger'].add_watch(self)
        except KeyError:
            raise Exception("Невозможно инициализировать нейрон без логгера!")

    def reset(self):
        super().reset()
        self.input_level = 0
        self.output = 0

    def hard_reset(self):
        super().hard_reset()
        self.reset()

    def update(self, synapse:str):
        pass

    def set_weight(self, weight:list):
        pass

    def mutate(self, *weights:list):
        pass



class STDPNeuron(Neuron):
    """Implementaion for a spike NN neuron"""

    def __init__(self, *args, **kwargs):
        """инициализация начальных параметров"""
        super().__init__(*args, **kwargs)
        try:
            param_set = kwargs['param_set']
        except KeyError:
            param_set = Defaults.param_set

        self.i_thres = param_set['i_thres']
        self.t_ltp = param_set['t_ltp']
        self.t_refrac = param_set['t_refrac']
        self.t_inhibit = param_set['t_inhibit']
        self.t_leak = param_set['t_leak']
        self.w_min = param_set['w_min']
        self.w_max = param_set['w_max']
        self.a_dec = param_set['a_dec']
        self.a_inc = param_set['a_inc']
        # self.b_dec     = param_set.b_dec
        # self.b_inc     = param_set.b_inc
        self.randmut = param_set['randmut']

        if 'activation_function' in kwargs:
            self.activation_function = kwargs['activation_function']
        else:
            self.activation_function = ActivationFunctions.DeltaFunction(self.i_thres)

        self.fired = False
        self.refractory = False
        self.ltp_synapses = {}  # записано, до какого момента при срабатывании синапс увеличит вес
        self.t_spike = 0
        self.t_last_spike = 0
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1

    def reset(self):
        super().reset()
        self.t_spike = -1
        self.t_last_spike = -1
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1
        self.ltp_synapses = {}

    def hard_reset(self):
        super().hard_reset()
        self.reset()

    def update(self, synapses: list):
        """обработать пришедшие данные и обновить состояние нейрона. Основная функция"""
        if self.clock <= self.inhibited_by:
            return
        self.t_last_spike, self.t_spike = self.t_spike, self.clock
        if self.t_last_spike != self.t_spike:
            # если на этом моменте времени ещё не обрабатывали
            self.output = 0

        self.refractory = False
        for synapse in synapses:
            self.input_level *= np.exp(-(self.t_spike - self.t_last_spike) / self.t_leak)
            self.input_level += self.weights[synapse]
            self.ltp_synapses[synapse] = self.clock + self.t_ltp
        self.output = self.activation_function(self.input_level)
        if self.output:
            self.input_level = 0
            self.refractory = True
            self.inhibited_by = self.t_spike + self.t_refrac
            if self.learn:
                for synapse, time in self.ltp_synapses.items():
                    if time >= self.t_spike - self.t_ltp:
                        self._synapse_inc_(synapse)
                    else:
                        self._synapse_dec_(synapse)
                self.ltp_synapses = {}
        return self.output

    def mutate(self, *weights: list):
        for synapse in self.weights:
            if random.random() > self.randmut:
                n = round((len(weights)-1)*random.random())
                self.weights[synapse] = weights[n][synapse]
            else:
                self.weights[synapse] = random.randrange(self.w_min, self.w_max)
            if self.weights[synapse] > self.w_max:
                self.weights[synapse] = self.w_max
            if self.weights[synapse] < self.w_min:
                self.weights[synapse] = self.w_min

    def _synapse_inc_(self, synapse: int):
        """усилить связь с синапсами, сработавшими прямо перед срабатыванием нейрона"""
        self.weights[synapse] += self.a_inc
        if self.weights[synapse] > self.w_max:
            self.weights[synapse] = self.w_max

    def _synapse_dec_(self, synapse: int):
        """ослабить связи с синапсами, не сработавшими перед срабатыванием нейрона"""
        self.weights[synapse] -= self.a_dec
        if self.weights[synapse] < self.w_min:
            self.weights[synapse] = self.w_min

    def inhibit(self):
        # эта функция вызывается только из модели в случае срабатывания другого нейрона
        # так как нейрон может быть неактивен, необходимо проверить сроки
        if (self.inhibited_on < self.clock) or not self.refractory:  # не надо дважды ингибировать в один фрейм
            self.inhibited_on = self.clock
            self.inhibited_by = self.clock + self.t_inhibit
        elif self.refractory and self.inhibited_by - self.t_refrac < self.inhibited_on:
            # в случае когда нейрон уже находится
            # в состоянии восстановления увеличиваем срок восстановления
            # но проверяем, не добавили ли к этому нейрону уже время ингибиции.
            # inhibited_by никогда не может превышать clock больше чем на сумму t_inhibit и t_refrac
            self.inhibited_by += self.t_inhibit

    def reset_input(self):
        self.input_level = 0

    def set_random_weights(self):
        self.weights = {k: random.randrange(self.w_min, self.w_max) for k in self.weights}

    def set_param_set(self, param_set):
        self.i_thres = param_set['i_thres']
        self.t_ltp = param_set['t_ltp']
        self.t_refrac = param_set['t_refrac']
        self.t_inhibit = param_set['t_inhibit']
        self.t_leak = param_set['t_leak']
        self.w_min = param_set['w_min']
        self.w_max = param_set['w_max']
        self.a_dec = param_set['a_dec']
        self.a_inc = param_set['a_inc']
        # self.b_dec     = param_set.b_dec
        # self.b_inc     = param_set.b_inc
        self.randmut = param_set['randmut']


def init_model(**kwargs):
    timer = Timer(0)

    feed_opts = {'timer': timer, 'mode': 'file', 'source': 'resources\\out.bin'}
    feed_opts.update(kwargs)
    feed = DataFeed(**feed_opts)

    model_opts = {'timer': timer, 'datafeed': feed, 'learn': True,
                  'param_set': pickle.load(open('optimal.param', 'rb'))['set'],
                  'logger': Logger(timer = timer)}
    model_opts.update(kwargs)

    model = SpikeNetwork(**model_opts)

    if 'teacher' in kwargs:
        model.logger.add_watch(model.teacher)

    return model, timer
