class Timer(object):
    """docstring for Timer"""

    def __init__(self, time=0):
        super(Timer, self).__init__()
        self.clock = time
        self.listeners = []

    def set_time(self, time):
        self.clock = time
        self.announce()

    def get_time(self):
        return self.clock

    def reset(self):
        self.set_time(0)
        for listener in self.listeners:
            listener.reset()

    def add_listener(self, listener):
        self.listeners.append(listener)

    def remove_listener(self, listeners):
        self.listeners.remove(listener)

    def announce(self):
        for listener in self.listeners:
            listener.update_clock(self.clock)
