from neuron import Neuron
from CameraFeed import CameraFeed

import matplotlib.pyplot as plt
import numpy as np

class Model(object):
	"""docstring for Model"""
	def __init__(self, feed:CameraFeed, pixels:tuple, structure:tuple):
		super(Model, self).__init__()
		
		self.neurons = []
		self.frame = 0
		self.feed = feed
		self.pixels = pixels
		self.genom = {pixels[x]:100 for x in range(len(pixels))}
		self.data_structure = {'length':40, 'time_length':22}
		self.structure = structure

		for layer_n in range(len(self.structure)):
			self.neurons.append([])
			for neuron in range(self.structure[layer_n]):
				self.add_neuron(self.genom, layer_n)

	def add_neuron(self, genom, l):
		self.neurons[l].append(Neuron(genom))

	def set_clock(self, time):
		self.clock = time

	def inhibit_neurons(self, fired_neuron):
		for n in range(self.neuron_number):
			neuron = self.neurons[n]
			if n != fired_neuron:
				neuron.inhibit(self.clock)

	def parse_aes(self, raw_data):
		synapse = raw_data>>self.data_structure['time_length']
		return (synapse,
			raw_data&0x3fffff)

	def tick(self):
		#добавить обработку работы сети по слоям с передачей вывода вглубь
			raw_data = self.feed.read()
		synapse, time = self.parse_aes(raw_data) #cutting bits 0:22
		self.set_clock(time) #устанавливаем часы по битам 22:0 входного пакета данных
		for n in range(self.neuron_number):
			neuron = self.neurons[n]
			neuron.update(self.clock, synapse)

			if neuron.has_fired():
				self.inhibit_neurons(n)

		self.frame += 1

	def get_journals(self):
		return [neuron.log() for neuron in self.neurons[-1]]

def plot_journals(m):
	journals = m.get_journals()
	fig = plt.figure()
	i=0
	for j in journals:
		i+=1
		ax = fig.add_subplot(len(journals), 1, i)
		xdata = [k for k in j]
		ydata = [j[k] for k in j]
		ax.plot(xdata, ydata, color='C'+str(i))
	plt.show()

if __name__ == "__main__":
	#(time+((polarty+((y_address+(x_address<<8))<<1))<<22))

	#feed = CameraFeed(mode='bnw', frames=100000, timescale=3)
	model = Model(feed, feed.get_pixels(), 2)
	while(1):
		try:
			model.tick()
		except:
			break
	plot_journals(model)
	
	