import random
import numpy as np
import torch

class Memory:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []
        self.count = 0

    def add(self, elem):
        self.count += 1
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
        self.buffer.append(elem)

    def get_mini_batch(self, size):
        return random.choices(self.buffer, k=size)

    def get_mini_batch_dim(self, size):
        rewards = []
        done = []
        states = []
        next = []
        actions = []
        for i in range(size):
            interaction = random.choice(self.buffer)
            states.append(interaction[0])
            actions.append(interaction[1])
            next.append(interaction[2])
            rewards.append(interaction[3])
            done.append(int(not interaction[4])) # If Done = False -> Agent done (used for computation of y in agent)
        return torch.Tensor(states), np.array(actions), torch.Tensor(next), torch.Tensor(rewards), torch.Tensor(done)