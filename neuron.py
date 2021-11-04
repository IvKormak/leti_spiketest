from utility import *
from timer import *

from math import exp
from random import gauss
IS_FIRED = True
NOT_FIRED = False


class Neuron(object):
    def __init__(self):
        pass


class STDPNeuron(Neuron):
    """Implementaion for a spike NN neuron"""

    def __init__(self, timer: Timer, genom, param_set=Defaults, learn=True, *args, **kwargs):
        """инициализация начальных параметров"""
        super(Neuron, self).__init__()
        timer.add_listener(self)

        self.i_thres = param_set.i_thres
        self.t_ltp = param_set.t_ltp
        self.t_refrac = param_set.t_refrac
        self.t_inhibit = param_set.t_inhibit
        self.t_leak = param_set.t_leak
        self.w_min = param_set.w_min
        self.w_max = param_set.w_max
        self.a_dec = param_set.a_dec
        self.a_inc = param_set.a_inc
        # self.b_dec     = param_set.b_dec
        # self.b_inc     = param_set.b_inc


        self.learn = learn

        for k in kwargs:
            self.__dict__[k] = kwargs[k]

        self.fired = False
        self.refractory = False
        self.input_level = 0
        self.event_journal = {}  # записаны аксоны, по которым пришли спайки, за tltp
        self.input_journal = []
        self.output_journal = []
        self.t_spike = 0
        self.t_last_spike = 0
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1
        self.clock = -1

        self._initiate_weights_(genom)  # адрес каждого веса задается битами 39:23 в входных данных

    def update_clock(self, time):
        self.clock = time

    def reset(self):
        self.t_spike = 0
        self.t_last_spike = 0
        self.inhibited_by = -1  # момент времени до начала симуляции
        self.inhibited_on = -1

    def _initiate_weights_(self, genom):
        self.weights = {k: gauss(genom[k], 20) for k in genom}

    def update(self, data: list):
        """обработать пришедшие данные и обновить состояние нейрона. Основная функция"""
        self.t_last_spike, self.t_spike = self.t_spike, self.clock
        if self.t_last_spike != self.t_spike:
            # если на этом моменте времени ещё не обрабатывали
            self.fired = False  # сбросить состояние срабатывания с предыдущего цикла обновления
        if self.clock > self.inhibited_by:
            self.refractory = False
            for synapse in data:
                self._spike_(synapse)

    def _spike_(self, synapse: int):
        """обработать пришедший импульс"""
        self.event_journal[self.t_spike] = synapse
        self.input_level = self.input_level * exp(-(self.t_spike - self.t_last_spike) / self.t_leak) + self.weights[
            synapse]

    def check_if_excited(self):
        """проверить не накопился ли потенциал действия выше порога и изменить веса аксонов"""
        self.input_journal.append(self.input_level)
        if self.input_level >= self.i_thres:
            self.output_journal.append(1)
            self.input_level = 0
            self.fired = True
            self._set_refractory_period_()
            for entry in self.event_journal:
                synapse = self.event_journal[entry]
                if int(entry) >= self.t_spike - self.t_ltp:
                    self._synapse_inc_(synapse)
                else:
                    self._synapse_dec_(synapse)
        else:
            self.output_journal.append(0)

    def _set_refractory_period_(self):
        # эта функция выполняется только когда нейрон активен и только что сработал
        # так как чтобы нейрон сработал он должен быть активен сравнение текущего
        # времени с сроком неактивности не производится
        self.refractory = True
        self.inhibited_by = self.t_spike + self.t_refrac

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

    def has_fired(self):
        """проверить не сработал ли нейрон"""
        return self.fired

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

    def get_input_journal(self):
        return self.input_journal

    def get_output_journal(self):
        return self.output_journal

    def get_genom(self):
        return self.weights

        