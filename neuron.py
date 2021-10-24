from math import exp
from random import gauss
IS_FIRED = True
NOT_FIRED = False

class Neuron(object):
	"""Implementaion for a spike NN neuron"""
	def __init__(self, genom):
		"""инициализация начальных параметров"""
		super(Neuron, self).__init__()
		self.i_thres = 700
		self.t_ltp = 2*10**1
		self.t_refrac = 10**2
		self.t_inhibit = 1.5*10**1
		self.t_leak = 5*10**1
		self.w_min = 1 #1+-02
		self.w_max = 1000 #1000+-200
		self.a_dec = 50#50+-10
		self.a_inc = 100#100+-20
		#self.b_dec = 0
		#self.b_inc = 0

		self.fired = NOT_FIRED
		self.input_level = 0
		self._initiate_weights_(genom) #адрес каждого веса задается битами 39:23 в входных данных
		self.event_journal = {} #записаны аксоны, по которым пришли спайки, за tltp
		self.input_journal = {}

		self.t_spike = 0
		self.t_last_spike = 0
		self.inhibited_by = -1

	def _initiate_weights_(self, genom):
		self.weights = {k: gauss(genom[k], 20) for k in genom}

	def get_genom(self):
		return self.weights

	def update(self, time, synapse):
		"""обработать пришедшие данные и обновить состояние нейрона. Основная функция"""
		self.input_journal[time] = self.input_level #записать историю изменения входного сигнала
		self.fired = NOT_FIRED #сбросить состояние срабатывания с предыдущего цикла обновления
		if time > self.inhibited_by:
			self.t_last_spike, self.t_spike = self.t_spike, time
			self._journal_spike_(synapse)
			self._spike_(synapse)
			if self._excited_():
				self._fire_()

	def _journal_spike_(self, synapse):
		"""записать пришедший спайк"""
		self.event_journal[self.t_spike] = synapse

	def _spike_(self, synapse: int):
		"""обработать пришедший импульс"""
		self.input_level = self.input_level*exp(-(self.t_spike-self.t_last_spike)/self.t_leak)+self.weights[synapse]

	def _excited_(self):
		"""проверить не накопился ли потенциал действия выше порога и изменить веса аксонов"""
		if self.input_level > self.i_thres:
			return True
		return False

	def _fire_(self):
		"""обработать срабатывание нейрона"""
		self.input_level = 0
		self.fired = IS_FIRED
		self._refrac_()
		increase_weights = [self.event_journal[entry] for entry in self.event_journal if int(entry) >= self.t_spike-self.t_ltp]
		for synapse in self.weights:
			if synapse in increase_weights:
				self._synapse_inc_(synapse)
			else:
				self._synapse_dec_(synapse)
		self._refrac_()

	def _refrac_(self):
		#эта функция выполняется только когда нейрон активен и только что сработал, поэтому ориентируется по внутреннему времени
		self.inhibited_by = self.t_spike + self.t_refrac

	def _synapse_inc_(self, synapse: int):
		"""усилить связь с аксоном, сработавшим прямо перед срабатыванием нейрона"""
		self.weights[synapse] += self.a_inc

	def _synapse_dec_(self, synapse: int):
		"""ослабить связи с аксонами, не сработавшими перед срабатыванием нейрона"""
		self.weights[synapse] -= self.a_dec

	def has_fired(self):
		"""проверить не сработал ли нейрон"""
		return self.fired

	def inhibit(self, time):
		#эта функция вызывается только из модели в случае срабатывания другого нейрона, поэтому ориентируется по внешнему времени
		if self.inhibited_by < time:
			self.inhibited_by = time + self.t_inhibit
		if self.inhibited_by >= time and self.inhibited_by - self.t_refrac < time: #проверяем, не добавили ли к этому нейрону уже время ингибиции. inhibited_by никогда не может превышать clock больше чем на сумму t_inhibit и t_refrac
			self.inhibited_by += self.t_inhibit #в случае когда нейрон уже находится 
				#в состоянии восстановления увеличиваем срок восстановления

	def log(self):
		return self.input_journal
		