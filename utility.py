def parse_aes(raw_data):
    data = []
    if isinstance(raw_data, str):
        raw_data = int(raw_data, base=16)
    if isinstance(raw_data, int):
        raw_data = [raw_data]
    for entry in raw_data:
        synapse = entry >> Defaults.time_bits
        synapse = synapse<<3
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
    t_leak = 5 * 10 ** 2
    w_min = 1  # 1+-02
    w_max = 1000  # 1000+-200
    a_dec = 50  # 50+-10
    a_inc = 150  # 100+-20
    b_dec = 0
    b_inc = 0

    UP = 1
    DOWN = 2
    LEFT = 3
    RIGHT = 4

    UPLEFT = 5
    UPRIGHT = 6
    DOWNLEFT = 7
    DOWNRIGHT = 8

    answers = [UP, RIGHT, DOWN, DOWNRIGHT, UP, UPRIGHT, UP, DOWN, DOWNLEFT, UPLEFT]