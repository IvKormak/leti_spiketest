import random, time

from main import *
import AERGen as ag


class PerceptronNeuron(Neuron):
    def __init__(self, model, output_address, inputs, learn=True, weights=None, mask=None):
        super(PerceptronNeuron, self).__init__(model, output_address, inputs, learn, weights, mask)
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
            self.ltp_times = {}
        return self.output_level


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
    time = card[0].time
    time_step = 1
    while card or offset >= 0:
        #print("time:", time)
        events = [event for event in card if event.time == time]
        if events and verbose:
            srow = ''.join(list(map(lambda x: str(x.polarity), events)))
            srow = srow.replace('0', '░')
            srow = srow.replace('1', '█')
            print(time, srow)
        card = card[len(events):]
        feed_events(model, title, events)
        if 1 in [model.state[o] for o in model.outputs] and verbose:
            print('spiked:', ' '.join([o for o in model.outputs if model.state[o]]))
        model.state = {_: 0 for _ in model.state}
        time += time_step
        if not card:
            offset -= time_step
    return time - 1


perceptron_inputs = [f'i{_}' for _ in range(1, 7)]
perceptron_outputs = list(rows.keys())

spike_inputs = perceptron_outputs
spike_outputs = [f's{_}' for _ in range(3)]

perceptron_nps = NeuronParametersSet(i_thres=320,
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

spike_nps = NeuronParametersSet(i_thres=700,
                                t_ltp=50,
                                t_refrac=400,
                                t_inhibit=50,
                                t_leak=100,
                                w_min=1,
                                w_max=255,
                                w_random=1,
                                a_inc=70,
                                a_dec=50,
                                activation_function="DeltaFunction")

gps = GeneralParametersSet(inhibit_radius=3,
                           epoch_length=50,
                           execution_thres=1,
                           terminate_on_epoch=3,
                           wta=1,
                           false_positive_thres=0.3,
                           mask=None)

afterburn = 0
cooldown = 150  

def ry_train():

    time_offset = 0
    m = Model(perceptron_nps, gps,
              state={_: 0 for _ in perceptron_inputs + perceptron_outputs + spike_outputs},
              layers=[],
              outputs=spike_outputs
              )

    trained_perceptron_layer = LayerStruct(shape=[7, 1], per_field_shape=[3, 1],
                                           neurons=[PerceptronNeuron(model=m,
                                                                     output_address=o,
                                                                     inputs=perceptron_inputs,
                                                                     learn=False,
                                                                     mask=gps.mask,
                                                                     weights=perceptron_weights[o])
                                                    for o in perceptron_outputs]
                                           )


    m.layers.append(trained_perceptron_layer)

    m.layers.append(LayerStruct(shape=[3, 1], per_field_shape=[7, 1],
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

    for neuron in m.layers[0].neurons:
        neuron.reset(soft=True)

    for neuron in m.layers[1].neurons:
        neuron.reset(soft=False)

    for label, neuron in zip(list(trained_weights.keys()), m.layers[1].neurons):
        neuron.label = label
        neuron.weights = trained_weights[label]['weights'].copy()
        neuron.train = False

    random.shuffle(test_cards)
    for neuron in m.layers[-1].neurons:
        neuron.learn = False
    for title, card in seq:
        card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
                in
                card]
        time_offset = feed_card(m, title, card, -1) + cooldown
    reset(m, issoft=True)
    random.shuffle(test_cards)
    for neuron in m.layers[-1].neurons:
        neuron.learn = False

    time_offset = 0
    for title, card in test_cards:
        card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
                in
                card]
        time_offset = feed_card(m, title, card, -1) + cooldown

    label_neurons(m, 3)

    return m

# seq = [random.choice(list(rows.keys())) for _ in range(gps.epoch_length)]
# seq = [(num, random.choice(rows[num])) for num in seq]
MI = 0
MA = 128
    
perceptron_weights = {
    'left-center': {'i1': MA, 'i2': MA, 'i3': MI, 'i4': MI, 'i5': MI, 'i6': MA},
    'center-right': {'i1': MI, 'i2': MA, 'i3': MA, 'i4': MA, 'i5': MI, 'i6': MI},
    'left-right': {'i1': MA, 'i2': MI, 'i3': MA, 'i4': MI, 'i5': MA, 'i6': MI},
    'left': {'i1': MA, 'i2': MI, 'i3': MI, 'i4': MI, 'i5': MA, 'i6': MA},
    'center':  {'i1': MI, 'i2': MA, 'i3': MI, 'i4': MA, 'i5': MI, 'i6': MA},
    'right': {'i1': MI, 'i2': MI, 'i3': MA, 'i4': MA, 'i5': MA, 'i6': MI},
    'all': {'i1': MA, 'i2': MA, 'i3': MA, 'i4': MI, 'i5': MI, 'i6': MI},
}

trained_weights = {
    
}
target = set(numbers.keys())
# seq = [random.choice(list(rows.keys())) for _ in range(gps.epoch_length)]
# seq = [(num, random.choice(rows[num])) for num in seq]

while not set(trained_weights.keys()) >= target:
    trained = [ry_train() for _ in range(10)]
    for model in trained:
        for neuron in model.layers[1].neurons:
            if -0.1 < neuron.error < 0.1:
                if neuron.label in trained_weights:
                    if neuron.error < trained_weights[neuron.label]['error']:
                        trained_weights[neuron.label] = {'weights': neuron.weights, 'error': neuron.error}
                else:
                    trained_weights[neuron.label] = {'weights': neuron.weights, 'error': neuron.error}
    print(trained_weights)


target = Model(perceptron_nps, gps,
               state={_: 0 for _ in perceptron_inputs + perceptron_outputs + spike_outputs},
               layers=[],
               outputs=spike_outputs
               )

trained_perceptron_layer = LayerStruct(shape=[7, 1], per_field_shape=[3, 1],
                                       neurons=[PerceptronNeuron(model=target,
                                                                 output_address=o,
                                                                 inputs=perceptron_inputs,
                                                                 learn=False,
                                                                 mask=gps.mask,
                                                                 weights=perceptron_weights[o])
                                                for o in perceptron_outputs]
                                       )

for w, n in zip(perceptron_weights.values(), trained_perceptron_layer.neurons):
    n.set_weights(w)

target.layers.append(trained_perceptron_layer)

target.layers.append(LayerStruct(shape=[3, 1], per_field_shape=[7, 1],
                                 neurons=[Neuron(model=target,
                                                 output_address=o,
                                                 inputs=spike_inputs,
                                                 learn=False,
                                                 mask=gps.mask,
                                                 neuron_parameters_set=spike_nps)
                                          for o in spike_outputs]
                                 )
                     )


labels = {k: 0 for k in numbers.keys()}
print(fill_model_from_pool(target, trained))

random.shuffle(test_cards)
time_offset = 0
for title, card in test_cards:
    card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time + time_offset) for e
            in
            card]
    time_offset = feed_card(target, title, card, 10, True) + cooldown

for neuron in target.layers[-1].neurons:
    if neuron.label:
        print(neuron.output_address, neuron.label, neuron.error)
        print(neuron.weights)
