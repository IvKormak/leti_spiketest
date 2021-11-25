from utility import *

class DataFeed:
    def __init__(self, *args, **kwargs):
        try:
            self.timer = kwargs['timer']
        except KeyError:
            raise Exception("Необходимо предоставить объект таймера!")
        self.data = ""

        self.cache = {}

        self.pixels = Defaults.pixels

        if 'source' in kwargs:
            self.load(kwargs['source'])
        if 'start' in kwargs:
            self.index = kwargs['start']
        else:
            self.index = 0

    def load(self, source:str):
        self.index = 0
        if source not in self.cache:
            file = open(source, 'r')
            self.data = ""
            hexadecimals = "0123456789abcdef"
            while char := file.read(1):
                if char in hexadecimals:
                    self.data += char
            self.cache[source] = self.data
        else:
            self.data = self.cache[source]

        self.timer.reset()

    def get_pixels(self):
        return self.pixels

    def __next__(self):
        try:
            entry = self.data[10*self.index:10*(self.index+1)]
        except IndexError:
            raise StopIteration
        if len(entry) == 0:
            raise StopIteration
        self.index += 1
        time = parse_aer(entry)[1]
        self.timer.set_time(time)
        return entry

    def __iter__(self):
        self.index = 0
        return self