from neuron import Neuron
from utility import *

class Layer(object):
    """docstring for Layer"""
    def __init__(self, timer:Timer, neuron_number, layer_number, genom, **kwargs):
        super(Layer, self).__init__()
        timer.add_listener(self)
        self.neuron_number = neuron_number
        self.neurons = [Neuron(timer=timer, genom=genom, **kwargs) for x in range(neuron_number)]
        self.layer_number = layer_number

    def update_clock(self, time):
        self.clock = time

    def tick(self, input_spikes):
        if type(input_spikes) == type(int()):
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

    def get_output(self):
        return list(map(
            lambda x: construct_aes(x, self.clock), 
            [
                (self.layer_number<<8)+(n<<1) 
                for n in range(len(self.neurons)) 
                if self.neurons[n].has_fired()
            ]
        ))

    def get_journals(self):
        return [neuron.log() for neuron in self.neurons]

    def get_genoms(self):
        return [neuron.get_genom() for neuron in self.neurons]
