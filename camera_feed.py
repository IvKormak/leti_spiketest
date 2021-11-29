from utility import *

class DataFeed:
    def __init__(self, *args, **kwargs):
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

        Timer.clock = 0

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
        time = self.parse_aer(entry)[1]
        Timer.clock = time
        return entry

    def __iter__(self):
        self.index = 0
        return self

    def parse_aer(self, raw_data):
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