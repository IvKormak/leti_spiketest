from neuron import *
from utility import *
from timer import *

from random import shuffle


class Layer(object):
    """docstring for Layer"""

    def __init__(self, timer: Timer, neuron_number, layer_number, genom, **kwargs):
        super(Layer, self).__init__()
        timer.add_listener(self)
        self.neuron_number = neuron_number
        self.neurons = [STDPNeuron(timer=timer, genom=genom, **kwargs) for _ in range(neuron_number)]
        self.layer_number = layer_number
        self.clock = -1

        if 'wta' in kwargs:
            self.wta = kwargs['wta']

    def reset(self):
        pass

    def get_synapses(self):
        return [
            format(self.layer_number, '02x')+
            format(n, '02x')
            for n in range(len(self.neurons))
        ]

    def get_fired_synapses(self):
        return [
                format(self.layer_number, '02x')+
                format(n, '02x')
                for n in range(len(self.neurons))
                if self.neurons[n].has_fired()
            ]

    def update_clock(self, time):
        self.clock = time

    def tick(self, input_spikes):
        if isinstance(input_spikes, int):
            input_spikes = list([input_spikes])
        for n in range(self.neuron_number):
            self.neurons[n].update(input_spikes)
        for n in range(self.neuron_number):
            self.neurons[n].check_if_excited()
            if self.neurons[n].has_fired():
                self.inhibit_neurons(n)

    def inhibit_neurons(self, fired_neuron):
        nns = list(range(self.neuron_number))
        shuffle(nns)
        for n in nns:
            neuron = self.neurons[n]
            if n != fired_neuron:
                neuron.inhibit()
                if self.wta:
                    neuron.reset_input()

    def get_input_journals(self):
        return [neuron.get_input_journal() for neuron in self.neurons]

    def get_output_journals(self):
        return [neuron.get_output_journal() for neuron in self.neurons]

    def get_genoms(self):
        return [neuron.get_genom() for neuron in self.neurons]
