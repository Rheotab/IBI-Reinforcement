import random


class Memory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, elem):
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(elem)

    def get_mini_batch(self, size):
        return random.choices(self.buffer, k=size)
