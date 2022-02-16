# This Python file uses the following encoding: utf-8

from collections import namedtuple
from typing import Union, Tuple, List
import numpy as np

Position = namedtuple('Position', ['x', 'y'])
Event = namedtuple('Event', ['address', 'position', 'polarity', 'time'])

TIME_BITS = 23
TIME_MASK = 0x7fffff

class AERGen:
    def __init__(self, radius:int = 1, speed:int = 100, pos_start:Position = Position(1,1), pos_end:Position = Position(28, 28), start_time:int = 0):
        if 0 > radius:
            raise ValueError("Diameter must be positive int!")
        if 0 > speed:
            raise ValueError("Speed must be positive int!")
        self.radius = radius
        try:
            phi = np.arctan((pos_end.y - pos_start.y)/(pos_end.x - pos_start.x))
            if pos_end.x < pos_start.x:
                phi = phi + np.pi
        except ZeroDivisionError:
            if pos_end.y > pos_start.y:
                phi = np.pi/2
            else:
                phi = -np.pi/2
        dx, dy = np.cos(phi)*radius, np.sin(phi)*radius
        self.pos_start = Position(pos_start.x-dx,
                                  pos_start.y-dy) #чтобы движение начиналось за краем экрана
        self.pos_end = Position(pos_end.x+dx,
                                pos_end.y+dy) #чтобы движение заканчивалось за краем экрана
        self.velocity = Position(np.cos(phi)*speed*10**-6, np.sin(phi)*speed*10**-6)

        self.total_time = np.sqrt(((self.pos_end.x - self.pos_start.x)**2+(self.pos_end.y - self.pos_start.y)**2)/(self.velocity.x**2+self.velocity.y**2))
        print(self.total_time)
        self.timestep = 0.1/speed*10**6
        self.time = start_time
        self.pixmap = np.zeros((28,28))>1
        self.frames = 0
        self.events = []

    def __next__(self):
        if not self.pixmap.any() and self.frames > 0:
            raise StopIteration
        if self.time > 2**23:
            raise ValueError("Timer overflow!")
        diff = self.diff_map(self.calculate_pixmap())
        while not diff:
            self.time += self.timestep
            diff = self.diff_map(self.calculate_pixmap())
            self.frames += 1
        self.pixmap = self.calculate_pixmap()
        return diff

    def __iter__(self):
        return self

    def calculate_pixmap(self):
        pos = Position(self.pos_start.x + self.velocity.x*self.time,
                       self.pos_start.y + self.velocity.y*self.time)
        if self.time > self.total_time:
            raise StopIteration
        return np.array([[np.sqrt((x-pos.x)**2+(y-pos.y)**2) for x in range(1,29)] for y in range(1,29)])<=self.radius

    def diff_map(self, map):
        diff = []
        for y in range(0,28):
            for x in range(0,28):
                if map[y][x] != self.pixmap[y][x]:
                    ev = Event(synapse_encode(x, y, 1), Position(x, y), 1 if map[y][x] else 0, round(self.time))
                    self.events.append(ev)
                    diff.append(ev)
        return diff

def aer_encode(event:Union[Event, List[Event]]):
    records = []
    if not isinstance(event, type([])):
        event = [event]
    for e in event:
        x = format(e.position.x, "02x")
        y = format(e.position.y, "02x")
        p = e.polarity << TIME_BITS
        t = e.time
        record = x+y+format(p+t, "06x")
        records.append(record)
    return records

def synapse_encode(x, y, p):
    return f"{format(x, '02x')}{format(y, '02x')}{p<<3}"


def aer_decode(entry:str):
    entry = int(entry.strip(), base = 16)
    synapse = entry >> TIME_BITS
    synapse = synapse << 3
    synapse = format(synapse, '05x')
    x = int(synapse[0:2], base=16)
    y = int(synapse[2:4], base=16)
    p = synapse[4] == '8'
    time = entry & TIME_MASK
    return Event(synapse, Position(x, y), p, time)

if __name__ == "__main__":
    test_trace = AERGen(radius = 3, speed = 5000)
    for d in test_trace:
        pass
    print("\n".join(aer_encode(test_trace.events)))
