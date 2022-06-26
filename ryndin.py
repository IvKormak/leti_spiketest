import random

from main import *
import AERGen as ag


class PerceptronNeuron(Neuron):
    def __init__(self, model, output_address, inputs, learn=True, weights=None, mask=None):
        super(PerceptronNeuron, self).__init__(model, output_address, inputs, learn, weights, mask)
        self.normalize()
        self.pre = {i: {'potential': 0, 'time': -1} for i in self.inputs}

    def inhibit(self):
        if self.output_level:
            self.inhibited_by = self.model.time + self.param_set.t_inhibit

    def random_weights(self):
        weights = {w: int(random.random() * (self.param_set.w_max - self.param_set.w_min) + self.param_set.w_min) for w
                   in self.inputs}
        return weights

    def reset(self, soft=False):
        self.pre = {_: {'potential': 0, 'time': -1} for _ in self.pre}
        super(PerceptronNeuron, self).reset(soft)

    def copy(self, model):
        return PerceptronNeuron(model, output_address=self.output_address, inputs=self.inputs, learn=self.learn)

    def update(self):
        inputs = self.inputs
        for address in inputs:
            if self.model.time - self.pre[address]['time'] > self.param_set.t_leak:
                self.pre[address]['potential'] = 0

        self.output_level = 0
        if self.t_last_spike == -1:
            self.t_spike = self.t_last_spike = self.model.time
        if self.model.time <= self.inhibited_by:
            return 0

        state = self.model.state  # копируем для ускорения доступа
        for address in inputs:
            if state[address]:
                self.pre[address]['potential'] = self.weights[address]
                self.ltp_times[address] = self.model.time + self.param_set.t_ltp
            self.pre[address]['time'] = self.model.time

        self.input_level = sum([p['potential'] for p in self.pre.values()])

        if self.param_set.activation_function == "DeltaFunction":
            self.output_level = int(self.input_level > self.param_set.i_thres)

        if self.output_level:
            self.times_fired += 1

            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn and self.age <= self.model.general_parameters_set.epoch_length:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                rest = [k for k in self.inputs if k not in not_rotten]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc if self.weights[synapse] > 0 else self.param_set.a_dec
                for synapse in rest:
                    self.weights[synapse] -= self.param_set.a_dec if self.weights[synapse] > 0 else self.param_set.a_inc
            self.normalize()
            self.ltp_times = {}
        return self.output_level

    def normalize(self):
        # if sum([x for x in self.weights.values() if x > 0]) > self.param_set.w_max:
        #     weights_scale = self.param_set.w_max / sum([x for x in self.weights.values() if x > 0])
        #     self.weights = {k: v * weights_scale if v > 0 else v for k, v in self.weights.items()}
        # if sum([x for x in self.weights.values() if x < 0]) < self.param_set.w_min:
        #     weights_scale = self.param_set.w_min / sum([x for x in self.weights.values() if x < 0])
        #     self.weights = {k: v * weights_scale if v < 0 else v for k, v in self.weights.items()}
        for synapse in self.weights:
            if self.weights[synapse] < self.param_set.w_min:
                self.weights[synapse] = self.param_set.w_min
            if self.weights[synapse] > self.param_set.w_max:
                self.weights[synapse] = self.param_set.w_max
            pairs = (['i1', 'i4'], ['i2', 'i5'], ['i3', 'i6'],)
            compl = [_ for _ in pairs if synapse in _][0]
            compl.remove(synapse)
            compl = compl[0]
            scale = self.weights[synapse] + self.weights[compl]
            if scale > self.param_set.w_max:
                self.weights[synapse] = int(self.weights[synapse] / scale * self.param_set.w_max)
            if scale < self.param_set.w_min:
                self.weights[synapse] = int(self.weights[synapse] / scale * self.param_set.w_min)


r = [ag.Event(address="i1", position=ag.Position(x=1, y=1), polarity=1, time=10),
     ag.Event(address="i2", position=ag.Position(x=2, y=1), polarity=1, time=10),
     ag.Event(address="i3", position=ag.Position(x=3, y=1), polarity=1, time=10),
     ag.Event(address="i4", position=ag.Position(x=1, y=1), polarity=0, time=10),
     ag.Event(address="i5", position=ag.Position(x=2, y=1), polarity=0, time=10),
     ag.Event(address="i6", position=ag.Position(x=3, y=1), polarity=0, time=10),
     ]


def construct_trace(pixels):
    events = [[r[p] if p + 1 in pset else r[p + 3] for p in range(3)] for pset in pixels]
    res = []
    dt = 10
    t = events[0][0].time
    for frame in events:
        res += [ag.Event(_.address, _.position, _.polarity, t) for _ in frame]
        t += dt
    return res


rows = {
    'left': [construct_trace([(1,)])],
    'center': [construct_trace([(2,)])],
    'right': [construct_trace([(3,)])],
    'left-center': [construct_trace([(1, 2)])],
    'center-right': [construct_trace([(2, 3)])],
    'left-right': [construct_trace([(1, 3)])],
    'all': [construct_trace([(1, 2, 3)])],
}
numbers = {
    'one': [
        construct_trace([
            (2,),
            (2,),
            (2,),
            (2,),
            (2,)
        ]),
        construct_trace([
            (1, 2,),
            (2,),
            (2,),
            (2,),
            (2,)
        ])
    ], 'two': [
        construct_trace([
            (1, 2, 3,),
            (3,),
            (1, 2, 3,),
            (1,),
            (1, 2, 3,)
        ]),
        construct_trace([
            (2, 3,),
            (3,),
            (1, 2, 3,),
            (1,),
            (1, 2, 3,)
        ])
    ], 'zero': [
        construct_trace([
            (1, 2, 3,),
            (1, 3,),
            (1, 3,),
            (1, 3,),
            (1, 2, 3,)
        ]),
        construct_trace([
            (1, 2, 3,),
            (1, 3,),
            (1, 3,),
            (3,),
            (1, 2, 3,)
        ])
    ],
}

test_rows = [(k, v[0]) for k, v in rows.items()] * 3
test_cards = [(k, v[0]) for k, v in numbers.items()] * 3


def feed_card(model, title, card, offset, verbose=False):
    if verbose:
        print("███████")
        print("███████")
    time = card[0].time
    time_step = 1
    visualize_trace(card)
    while card or offset >= 0:
        print("time:", time)
        events = [event for event in card if event.time == time]
        if events and verbose:
            srow = ''.join(list(map(lambda x: str(x.polarity), events)))
            srow = srow.replace('0', '░')
            srow = srow.replace('1', '█')
            print(time, srow)
        card = card[len(events):]
        feed_events(model, title, events)
        if 1 in [model.state[o] for o in outputs]:
            print('spiked:', ' '.join([o for o in outputs if model.state[o]]))
        model.state = {_: 0 for _ in model.state}
        time += time_step
        if not card:
            offset -= time_step
    return time - 1


perceptron_inputs = [f'i{_}' for _ in range(1, 7)]
perceptron_outputs = [f'o{_}' for _ in range(1, 10)]

spike_inputs = perceptron_outputs
spike_outputs = [f's{_}' for _ in range(3)]

perceptron_nps = NeuronParametersSet(i_thres=128,
                                     t_ltp=5,
                                     t_refrac=0,
                                     t_inhibit=0,
                                     t_leak=5,
                                     w_min=-127,
                                     w_max=128,
                                     w_random=1,
                                     a_inc=15,
                                     a_dec=5,
                                     activation_function="DeltaFunction")

spike_nps = NeuronParametersSet(i_thres=1400,
                                t_ltp=50,
                                t_refrac=100,
                                t_inhibit=30,
                                t_leak=100,
                                w_min=1,
                                w_max=255,
                                w_random=1,
                                a_inc=15,
                                a_dec=10,
                                activation_function="DeltaFunction")

gps = GeneralParametersSet(inhibit_radius=3,
                           epoch_length=50,
                           execution_thres=1,
                           terminate_on_epoch=3,
                           wta=0,
                           false_positive_thres=0.3,
                           mask=None)

# seq = [random.choice(list(rows.keys())) for _ in range(gps.epoch_length)]
# seq = [(num, random.choice(rows[num])) for num in seq]
MI = -127
MA = 128
weights = [
    {'i1': MI, 'i2': MI, 'i3': MA, 'i4': MA, 'i5': MA, 'i6': MI},
    {'i1': MI, 'i2': MA, 'i3': MI, 'i4': MA, 'i5': MI, 'i6': MA},
    {'i1': MA, 'i2': MI, 'i3': MI, 'i4': MI, 'i5': MA, 'i6': MA},
    {'i1': MI, 'i2': MA, 'i3': MA, 'i4': MA, 'i5': MI, 'i6': MI},
    {'i1': MA, 'i2': MI, 'i3': MA, 'i4': MI, 'i5': MA, 'i6': MI},
    {'i1': MA, 'i2': MA, 'i3': MI, 'i4': MI, 'i5': MI, 'i6': MA},
    {'i1': MA, 'i2': MA, 'i3': MA, 'i4': MI, 'i5': MI, 'i6': MI},
]


def ry_train():
    #    test_cards = test_rows
    time_offset = 0
    afterburn = 0
    cooldown = 150

    m = Model(perceptron_nps, gps,
              state={_: 0 for _ in perceptron_inputs + perceptron_outputs + spike_outputs},
              layers=[],
              outputs=spike_outputs
              )

    trained_perceptron_layer = LayerStruct(shape=[9, 1], per_field_shape=[3, 1],
                                           neurons=[PerceptronNeuron(model=m,
                                                                     output_address=o,
                                                                     inputs=perceptron_inputs,
                                                                     learn=False,
                                                                     mask=gps.mask)
                                                    for o in perceptron_outputs]
                                           )

    for w, n in zip(weights, trained_perceptron_layer.neurons):
        n.set_weights(w)

    m.layers.append(trained_perceptron_layer)

    m.layers.append(LayerStruct(shape=[3, 1], per_field_shape=[9, 1],
                                neurons=[Neuron(model=m,
                                                output_address=o,
                                                inputs=spike_inputs,
                                                learn=True,
                                                mask=gps.mask,
                                                neuron_parameters_set=spike_nps)
                                         for o in spike_outputs]
                                )
                    )

    seq = [random.choice(list(numbers.keys())) for _ in range(gps.epoch_length)]
    seq = [(num, random.choice(numbers[num])) for num in seq]

trained_weights = {
    'left-center': {'i1': 46, 'i2': 62, 'i3': -57, 'i4': -75, 'i5': -66, 'i6': 71},
    'center-right': {'i1': -14, 'i2': 50, 'i3': 100, 'i4': 124, 'i5': -70, 'i6': -69},
    'left-right': {'i1': 114, 'i2': -70, 'i3': 61, 'i4': -69, 'i5': 100, 'i6': -49},
    'left': {'i1': 114, 'i2': 4, 'i3': -61, 'i4': -69, 'i5': 60, 'i6': 60},
    'center': {'i1': -46, 'i2': 62, 'i3': -57, 'i4': 75, 'i5': -66, 'i6': 71},
    'right': {'i1': -37, 'i2': -90, 'i3': 39, 'i4': 70, 'i5': 87, 'i6': -62},
    'all': {'i1': 114, 'i2': 111, 'i3': 61, 'i4': -69, 'i5': -70, 'i6': -49},
}
target = set(rows.keys())
# seq = [random.choice(list(rows.keys())) for _ in range(gps.epoch_length)]
# seq = [(num, random.choice(rows[num])) for num in seq]
test_cards = test_rows
time_offset = 0
afterburn = 0
cooldown = 150

while set(trained_weights.keys()) != target:
    print('*')
    for title, card in seq:
        card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e in
                card]
        time_offset = feed_card(m, card, afterburn) + cooldown
    reset(m, issoft=False)
    if trained_weights:
        for label, neuron in zip(list(trained_weights.keys()), m.layers[0].neurons):
            neuron.label = label
            neuron.weights = trained_weights[label][0].copy()
            neuron.train = False

    random.shuffle(test_cards)
    for neuron in m.layers[-1].neurons:
        neuron.learn = False
    for title, card in seq:
        card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
                in
                card]
        time_offset = feed_card(m, title, card, afterburn) + cooldown
    reset(m, issoft=True)
    random.shuffle(test_cards)
    for neuron in m.layers[-1].neurons:
        neuron.learn = False

    time_offset = 0
    for title, card in test_cards:
        card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
                in
                card]
        time_offset = feed_card(m, title, card, afterburn) + cooldown

    label_neurons(m, 3)
    return m


target = Model(perceptron_nps, gps,
               state={_: 0 for _ in perceptron_inputs + perceptron_outputs + spike_outputs},
               layers=[],
               outputs=spike_outputs
               )

trained_perceptron_layer = LayerStruct(shape=[9, 1], per_field_shape=[3, 1],
                                       neurons=[PerceptronNeuron(model=target,
                                                                 output_address=o,
                                                                 inputs=perceptron_inputs,
                                                                 learn=False,
                                                                 mask=gps.mask)
                                                for o in perceptron_outputs]
                                       )

for w, n in zip(weights, trained_perceptron_layer.neurons):
    n.set_weights(w)

target.layers.append(trained_perceptron_layer)

target.layers.append(LayerStruct(shape=[3, 1], per_field_shape=[9, 1],
                                 neurons=[Neuron(model=target,
                                                 output_address=o,
                                                 inputs=spike_inputs,
                                                 learn=False,
                                                 mask=gps.mask,
                                                 neuron_parameters_set=spike_nps)
                                          for o in spike_outputs]
                                 )
                     )

trained = [ry_train() for _ in range(10)]
labels = {k: 0 for k in numbers.keys()}
print(fill_model_from_pool(target, trained))

random.shuffle(test_cards)
time_offset = 0
for title, card in test_cards:
    card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
            in
            card]
    time_offset = feed_card(target, title, card, 10, True)

for neuron in target.layers[-1].neurons:
    if neuron.label:
        print(neuron.output_address, neuron.label, neuron.error)
        print(neuron.weights)
