import random

from main import *
import AERGen as ag

class FlushNeuron(Neuron):
    def __init__(self, *args, **kwargs):
        super(FlushNeuron, self).__init__(*args, **kwargs)
        self.pre = {i: {'potential': 0, 'time': -1} for i in self.inputs}

    def reset(self, soft=False):
        self.pre = {_: {'potential': 0, 'time': -1} for _ in self.pre}
        super(FlushNeuron, self).reset(soft)

    def copy(self, model):
        return FlushNeuron(model, output_address=self.output_address, inputs=self.inputs, learn=self.learn)

    def update(self):
        inputs = self.inputs
        for address in inputs:
            self.pre[address]['potential'] = np.exp(
                -(self.model.time - self.pre[address]['time']) / self.param_set.t_leak)

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

        for address in self.pre:
            self.input_level = sum([p['potential'] for p in self.pre.values()])

        if self.param_set.activation_function == "DeltaFunction":
            self.output_level = int(self.input_level > self.param_set.i_thres)
        if self.output_level:
            self.times_fired += 1

            min_level = self.param_set.w_min
            self.input_level = 0
            self.model.state[self.output_address] = 1
            self.inhibited_by = self.t_spike + self.param_set.t_refrac
            if self.learn and self.age <= self.model.general_parameters_set.epoch_length:
                not_rotten = [k for k in self.ltp_times.keys() if
                              self.ltp_times[k] >= self.t_spike - self.param_set.t_ltp]
                rest = [k for k in self.inputs if k not in not_rotten]
                for synapse in not_rotten:
                    self.weights[synapse] += self.param_set.a_inc
                    if self.weights[synapse] > self.weights_mask[synapse]:
                        self.weights[synapse] = self.weights_mask[synapse]
                for synapse in rest:
                    self.weights[synapse] -= self.param_set.a_dec
                    if self.weights[synapse] < min_level:
                        self.weights[synapse] = min_level
            self.ltp_times = {}
        return self.output_level

r = [ag.Event(address="i1", position=ag.Position(x=1, y=1), polarity=1, time=0),
     ag.Event(address="i2", position=ag.Position(x=2, y=1), polarity=1, time=0),
     ag.Event(address="i3", position=ag.Position(x=3, y=1), polarity=1, time=0)
     ]


def construct_trace(events):
    res = []
    dt = 10
    t = events[0][0].time
    for frame in events:
        res += [ag.Event(_.address, _.position, _.polarity, t) for _ in frame]
        t += dt
    return res


numbers = {
    "one": [
        construct_trace([
            [r[1]],
            [r[1]],
            [r[1]],
            [r[1]],
            [r[1]],
        ]),
        construct_trace([
            [r[0], r[1]],
            [r[1]],
            [r[1]],
            [r[1]],
            [r[1]],
        ])
    ],
    "two": [
        construct_trace([
            [r[0],r[1],r[2]],
            [r[2]],
            [r[0],r[1],r[2]],
            [r[0]],
            [r[0],r[1],r[2]],
        ]),
        construct_trace([
            [r[1],r[2]],
            [r[2]],
            [r[0],r[1],r[2]],
            [r[0]],
            [r[0],r[1],r[2]],
        ])
    ],
    "zero": [
        construct_trace([
            [r[0],r[1],r[2]],
            [r[0], r[2]],
            [r[0], r[2]],
            [r[0], r[2]],
            [r[0],r[1],r[2]],
        ]),
        construct_trace([
            [r[1]],
            [r[0], r[2]],
            [r[0], r[2]],
            [r[0], r[2]],
            [r[0],r[1],r[2]],
        ])
    ]
}

test_cards = [
    ("one", construct_trace([
        [r[1]],
        [r[2]],
        [r[1]],
        [r[1]],
        [r[1]],
    ])),
    ("one", construct_trace([
        [r[0]],
        [r[1]],
        [r[1]],
        [r[1]],
        [r[1]],
    ])),
    ("two", construct_trace([
        [r[0],r[1],r[2]],
        [],
        [r[0],r[1],r[2]],
        [r[0]],
        [r[0],r[1],r[2]],
    ])),
    ("two", construct_trace([
        [r[0],r[1], r[2]],
        [r[2]],
        [r[0],r[1],r[2]],
        [r[0]],
        [r[0],r[1]],
    ])),
    ("zero", construct_trace([
        [r[0],r[1],r[2]],
        [r[0], r[2]],
        [r[0]],
        [r[0], r[2]],
        [r[0],r[1],r[2]],
    ])),
    ("zero", construct_trace([
        [r[0], r[1], r[2]],
        [r[0], r[2]],
        [r[0], r[2]],
        [r[0], r[2]],
        [r[0],r[1]],
    ]))
]*3

def visualize_trace(trace):
    print("num: || time:")
    while trace:
        time = trace[0].time
        buf = [(-1*(_.polarity==0)+1*_.polarity)*(_.position.x) for _ in trace if _.time == time]
        trace = trace[len(buf):]
        s = ""
        for x in range(1,4):
            if x in buf:
                s += '*'
            elif -1*x in buf:
                s += '_'
            else:
                s += ' '
        print(s, " ||", time)

def feed_card(model, card):
    print("="*10)
    time = card[0].time
    time_step = 1
    visualize_trace(card)
    while card:
        print("time:", time)
        events = [event for event in card if event.time == time]
        card = card[len(events):]
        feed_events(model, title, events)
        if 1 in [model.state[o] for o in outputs]:
            print('spiked:', ' '.join([o for o in outputs if model.state[o]]))
        model.state = {_:0 for _ in model.state}
        time += time_step
    return time-1


inputs = [f'i{_}' for _ in range(3)]
outputs = [f'o{_}' for _ in range(9)]

nps = NeuronParametersSet(i_thres=250,
                          t_ltp=30,
                          t_refrac=100,
                          t_inhibit=20,
                          t_leak=50,
                          w_min=1,
                          w_max=255,
                          w_random=1,
                          a_inc=10,
                          a_dec=2,
                          activation_function="DeltaFunction")
gps = GeneralParametersSet(inhibit_radius=1,
                           epoch_length=20,
                           execution_thres=1,
                           terminate_on_epoch=3,
                           wta=0,
                           false_positive_thres=1,
                           mask=None)

m = Model(nps, gps,
          state={_: 0 for _ in inputs + outputs},
          layers=[],
          outputs=outputs
          )

m.layers.append(LayerStruct(shape=[9, 1], per_field_shape=[3, 1],
                            neurons=[FlushNeuron(m, o, inputs, True, mask=gps.mask)
                                     for o in outputs]
                            )
                )


seq = [random.choice(list(numbers.keys())) for _ in range(gps.epoch_length)]
seq = [(num, random.choice(numbers[num])) for num in seq]
time_offset = 0
cooldown = 0

for title, card in seq:
    card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time+time_offset) for e in card]
    time_offset = feed_card(m, card) + cooldown

reset(m, issoft=True)
random.shuffle(test_cards)
for neuron in m.layers[-1].neurons:
    neuron.learn = False

time_offset = 0
for title, card in test_cards:
    card = [ag.Event(address=e.address, position=e.position, polarity=e.polarity, time=e.time+time_offset) for e in card]
    time_offset = feed_card(m, card) + cooldown

label_neurons(m, len(test_cards)/3)
for neuron in m.layers[-1].neurons:
    print(neuron.output_address, neuron.label, neuron.error)
    print(neuron.weights)




