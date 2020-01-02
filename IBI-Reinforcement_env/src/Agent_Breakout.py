import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
import random
from gym import wrappers, logger
from DQN_Breakout import Net
from Memory import Memory
import numpy as np
import time
import copy


class Agent(object):
    def __init__(self, nb_ep, action_space, buffer_size=100000, epsilon=0.1, batch_size=50, gamma=0.8, eta=0.005,
                 N=100):
        self.action_space = action_space
        self.eps = epsilon  # e-greedy
        self.batch_size = batch_size  # what do we learn
        self.gamma = gamma  # how much is the importance of reward in learning
        self.eta = eta  # learning rate
        self.N = N  # When do we transfer to target -> Improve stability
        self.count_N = 0
        self.memory = Memory(buffer_size)
        self.qlearning_nn = Net(64)  # FIXME
        self.target_network = Net(64)  # FIXME
        self.target_network.load_state_dict(self.qlearning_nn.state_dict())
        self.optimiser = torch.optim.Adam(self.qlearning_nn.parameters(), lr=self.eta)
        self.arr_loss = []

    def act(self, observation, reward, done):
        qvalues = self.qlearning_nn(torch.Tensor(observation))
        value = self.politique_greedy(qvalues)

        return int(value)

    def action_transform_index_into_valid(self, index):
        if index == 2:
            return 2
        if index == 3:
            return 3
        return int(index)

    def memorise(self, interaction):
        self.memory.add(interaction)

    def politique_greedy(self, qval):
        qval_np = qval.clone().detach().numpy()
        if random.random() < self.eps:
            return self.action_space.sample()
        a = np.array([])
        a = np.append(a, np.argmax(qval_np))
        return np.random.choice(a)


    # FIXME index
    def politique_boltzmann(self, qval, tau):
        # TODO
        pass

    def learn(self):
        self.count_N += 1
        minibatch = self.memory.get_mini_batch(self.batch_size)
        for interaction in minibatch:
            state = torch.Tensor(interaction[0])
            state_next = torch.Tensor(interaction[2])
            qvalues = self.qlearning_nn(state)
            action = interaction[1]
            reward = interaction[3]
            qval_prec = qvalues[action]
            if interaction[4]:
                tmp = torch.Tensor([reward]).reshape(1)
            else:
                qvalues_next = self.target_network(state_next)
                qmax = torch.max(qvalues_next)
                tmp = torch.Tensor([reward + self.gamma * qmax]).reshape(1)
            loss = F.mse_loss(qval_prec.reshape(1), tmp)
            self.arr_loss.append(loss)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        if self.N == self.count_N:
            self.count_N = 0
            self.target_network.load_state_dict(self.qlearning_nn.state_dict())

    def show_loss(self):
        plt.plot(self.arr_loss)
        plt.show()

    def save_weights(self, txt):
        torch.save(self.target_network.state_dict(), txt)
