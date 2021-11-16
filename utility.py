def parse_aes(raw_data):
    data = []
    time = 0
    if isinstance(raw_data, str):
        raw_data = int(raw_data, base=16)
    if isinstance(raw_data, int):
        raw_data = [raw_data]
    for entry in raw_data:
        synapse = entry >> Defaults.time_bits
        synapse = synapse << 3
        synapse = format(synapse, '05x')
        time = entry & Defaults.time_mask
        # print(format(raw_data, "#040b"))
        # print(synapse,
        #    raw_data&defaults.time_mask)
        data.append(synapse)
    return data, time


def construct_aes(synapse, time):
    return (synapse << Defaults.time_bits) + time


class Defaults(object):
    """docstring for defaults"""
    time_bits = 23
    time_mask = 0x3fffff
    data_bits = 40

    i_thres = 1000
    t_ltp = 3 * 10 ** 2
    t_refrac = 10 ** 3
    t_inhibit = 1.5 * 10 ** 2
    t_leak = 4 * 10 ** 2
    w_min = 1  # 1+-02
    w_max = 1000  # 1000+-200
    a_dec = 100  # 50+-10
    a_inc = 300  # 100+-20
    b_dec = 0
    b_inc = 0
    mutation_const = 10
    randmut = 0.01

    UP = 1
    DOWN = 2
    LEFT = 3
    UPLEFT = 4
    UPRIGHT = 5
    DOWNLEFT = 6
    DOWNRIGHT = 7

    files = ["trace_up.bin",
             "trace_down.bin",
             "trace_right.bin",
             "trace_upleft.bin",
             "trace_upright.bin",
             "trace_downleft.bin",
             "trace_downright.bin"]