import random
from typing import Type

from main import *
import AERGen as ag

SimpleEvent = namedtuple("SimpleEvent", ["address", "time"])


class PerceptronNeuron(Neuron):
    def __init__(self, model, output_address, inputs, learn=True, weights=None, mask=None):
        super(PerceptronNeuron, self).__init__(model, output_address, inputs, learn, weights, mask)
        self.normalize()
        self.pre = {i: {'potential': 0, 'time': -1} for i in self.inputs}

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

row_events = {
    'left': SimpleEvent(address='left', time=0),
    'center': SimpleEvent(address='center', time=0),
    'right': SimpleEvent(address='right', time=0),
    'left-center': SimpleEvent(address='left-center', time=0),
    'left-right': SimpleEvent(address='left-right', time=0),
    'center-right': SimpleEvent(address='center-right', time=0),
    'all': SimpleEvent(address='all', time=0),
}

numbers_from_rows = {
    'one': [
        construct_trace([
            row_events['center'],
            row_events['center'],
            row_events['center'],
            row_events['center'],
            row_events['center']
        ]),
        construct_trace([
            row_events['left-center'],
            row_events['center'],
            row_events['center'],
            row_events['center'],
            row_events['center']
        ])
    ], 'two': [
        construct_trace([
            row_events['center-right'],
            row_events['right'],
            row_events['all'],
            row_events['left'],
            row_events['all']
        ]),
        construct_trace([
            row_events['all'],
            row_events['right'],
            row_events['all'],
            row_events['left'],
            row_events['all']
        ])
    ], 'zero': [
        construct_trace([
            row_events['all'],
            row_events['left-right'],
            row_events['left-right'],
            row_events['left-right'],
            row_events['all']
        ]),
        construct_trace([
            row_events['all'],
            row_events['left-right'],
            row_events['left-right'],
            row_events['left'],
            row_events['all']
        ])
    ],
}

test_rows = [(k, v[0]) for k, v in rows.items()] * 3
test_cards = [(k, v[0]) for k, v in numbers_from_rows.items()] * 3


def feed_card(model, title, card, offset):
    time = card[0].time
    time_step = 1
    while card or offset >= 0:
        events = [event for event in card if event.time == time]
        card = card[len(events):]
        feed_events(model, title, events)
        model.state = {_: 0 for _ in model.state}
        time += time_step
        if not card:
            offset -= time_step
    return time - 1


perceptron_inputs = [f'i{_}' for _ in range(1, 7)]
perceptron_outputs = [f'o{_}' for _ in range(1, 10)]

spike_inputs = perceptron_outputs
spike_outputs = [f's{_}' for _ in range(3)]

perceptron_nps = NeuronParametersSet(i_thres=125,
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
perceptron_gps = GeneralParametersSet(inhibit_radius=3,
                                      epoch_length=50,
                                      execution_thres=1,
                                      terminate_on_epoch=3,
                                      wta=0,
                                      false_positive_thres=0.3,
                                      mask=None)

spike_nps = NeuronParametersSet(i_thres=125,
                                t_ltp=50,
                                t_refrac=100,
                                t_inhibit=30,
                                t_leak=100,
                                w_min=-127,
                                w_max=128,
                                w_random=1,
                                a_inc=15,
                                a_dec=5,
                                activation_function="DeltaFunction")
spike_gps = GeneralParametersSet(inhibit_radius=3,
                                 epoch_length=50,
                                 execution_thres=1,
                                 terminate_on_epoch=3,
                                 wta=0,
                                 false_positive_thres=0.3,
                                 mask=None)


# seq = [random.choice(list(rows.keys())) for _ in range(gps.epoch_length)]
# seq = [(num, random.choice(rows[num])) for num in seq]

weights = [
    {'i1': -127, 'i2': 128, 'i3': -122, 'i4': 128, 'i5': -127, 'i6': 123},
    {'i1': 128, 'i2': 41, 'i3': 128, 'i4': -127, 'i5': 87, 'i6': -127},
    {'i1': 82, 'i2': -117, 'i3': 128, 'i4': 46, 'i5': 128, 'i6': -117},
    {'i1': 128, 'i2': 128, 'i3': 63, 'i4': -127, 'i5': -122, 'i6': 65},
    {'i1': 84, 'i2': 128, 'i3': 128, 'i4': 44, 'i5': -117, 'i6': -122},
    {'i1': 128, 'i2': 41, 'i3': 128, 'i4': -127, 'i5': 87, 'i6': -127},
    {'i1': -127, 'i2': 35, 'i3': 128, 'i4': 128, 'i5': 93, 'i6': -127},
    {'i1': 82, 'i2': -117, 'i3': 128, 'i4': 46, 'i5': 128, 'i6': -117},
    {'i1': 128, 'i2': -117, 'i3': -128, 'i4': 46, 'i5': 70, 'i6': 117},
]

def ry_train():
#    test_cards = test_rows
    time_offset = 0
    afterburn = 0
    cooldown = 150

    m = Model(perceptron_nps, perceptron_gps,
              state={_: 0 for _ in perceptron_inputs + perceptron_outputs + spike_outputs},
              layers=[],
              outputs=spike_outputs
              )

    trained_perceptron_layer = LayerStruct(shape=[9, 1], per_field_shape=[3, 1],
                                     neurons=[PerceptronNeuron(model=m,
                                                               output_address=o,
                                                               inputs=perceptron_inputs,
                                                               learn=False,
                                                               mask=perceptron_gps.mask)
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
                                                mask=spike_gps.mask)
                                         for o in spike_outputs]
                                )
                    )

    seq = [random.choice(list(numbers_from_rows.keys())) for _ in range(perceptron_gps.epoch_length)]
    seq = [(num, random.choice(numbers_from_rows[num])) for num in seq]

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


target = Model(perceptron_nps, perceptron_gps,
               state={_: 0 for _ in perceptron_inputs + perceptron_outputs},
               layers=[],
               outputs=perceptron_outputs
               )

trained_perceptron_layer = LayerStruct(shape=[9, 1], per_field_shape=[3, 1],
                                       neurons=[PerceptronNeuron(model=target,
                                                                 output_address=o,
                                                                 inputs=perceptron_inputs,
                                                                 learn=False,
                                                                 mask=perceptron_gps.mask)
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
                                            mask=spike_gps.mask)
                                     for o in spike_outputs]
                            )
                )

trained = [ry_train() for _ in range(10)]
labels = {k: 0 for k in row_events.keys()}
print(fill_model_from_pool(target, trained))

for neuron in target.layers[-1].neurons:
    if neuron.label:
        print(neuron.output_address, neuron.label, neuron.error)
        print(neuron.weights)
