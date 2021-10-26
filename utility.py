def parse_aes(raw_data):
    data = []
    if type(raw_data) == type(int()):
        raw_data = list([raw_data])
    for entry in raw_data:
        synapse = entry>>defaults.time_bits
        time = entry&defaults.time_mask
        #print(format(raw_data, "#040b"))
        #print(synapse,
        #    raw_data&defaults.time_mask)
        data.append(synapse)
    return data, time

def construct_aes(synapse, time):
    return (synapse<<defaults.time_bits)+time

class defaults(object):
    """docstring for defaults"""
    time_bits = 22
    time_mask = 0x3fffff
    data_bits = 40
    #for 16*16 pixel grid
    i_thres     = 400
    t_ltp       = 2*10**2
    t_refrac    = 10**3
    t_inhibit   = 1.5*10**2
    t_leak      = 5*10**2
    w_min       = 1 #1+-02
    w_max       = 1000 #1000+-200
    a_dec       = 50#50+-10
    a_inc       = 100#100+-20
    b_dec       = 0
    b_inc       = 0
    

class Timer(object):
    """docstring for Timer"""
    def __init__(self, time=0):
        super(Timer, self).__init__()
        self.clock = time
        self.listeners = []

    def set_time(self,time):
        self.clock = time
        self.announce()

    def get_time(self):
        return time

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listeners):
        self.listeners.remove(listener)

    def announce(self):
        for listener in self.listeners:
            listener.update_clock(self.clock)
        