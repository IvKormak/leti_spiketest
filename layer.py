from neuron import *
from utility import *


class Layer(object):
    """docstring for Layer"""

    def __init__(self, timer: Timer, neuron_number, layer_number, genom, **kwargs):
        super(Layer, self).__init__()
        timer.add_listener(self)
        self.neuron_number = neuron_number
        self.neurons = [STDPNeuron(timer=timer, genom=genom, **kwargs) for _ in range(neuron_number)]
        self.layer_number = layer_number
        self.clock = -1

    def get_synapses(self):
        return [
            (self.layer_number << 8) + (n << 1)
            for n in range(len(self.neurons))
        ]

    def get_fired_synapses(self):
        return [
                (self.layer_number << 8) + (n << 1)
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
        for n in range(self.neuron_number):
            neuron = self.neurons[n]
            if n != fired_neuron:
                neuron.inhibit()

    def get_journals(self):
        return [neuron.log() for neuron in self.neurons]

    def get_genoms(self):
        return [neuron.get_genom() for neuron in self.neurons]
