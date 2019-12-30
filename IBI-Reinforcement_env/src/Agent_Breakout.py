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
    def __init__(self,nb_ep, action_space, buffer_size=10000, epsilon=0.3, batch_size=10, gamma=0.05, eta=0.005, N = 100):
        self.action_space = action_space
        self.eps = epsilon # e-greedy
        self.batch_size = batch_size # what do we learn
        self.gamma = gamma # how much is the importance of reward in learning
        self.eta = eta # learning rate
        self.N = N # When do we transfer to target -> Improve stability
        self.count_N = 0
        self.memory = Memory(buffer_size)
        self.qlearning_nn = Net(64) # FIXME
        self.target_network = Net(64) # FIXME
        self.target_network.load_state_dict(self.qlearning_nn.state_dict())
        self.optimiser = torch.optim.Adam(self.qlearning_nn.parameters(), lr=self.eta)
        self.arr_loss = []



    def act(self, observation, reward, done):
        qvalues = self.qlearning_nn(torch.Tensor(observation).reshape(1,4)) # fixme reshape
        value = self.politique_greedy(qvalues)
        return value

    def memorise(self, interaction):
        self.memory.add(interaction)

    def politique_greedy(self, qval):
        qval_np = qval.clone().detach().numpy()
        if random.random() < self.eps:
            return np.where(qval_np[0] == np.random.choice(qval_np[0], size=1))[0][0]
        return np.argmax(qval_np[0])

    # FIXME index
    def politique_boltzmann(self, qval, tau):
        qval_np = qval.detach().numpy()
        s = 0
        prob = np.array([])
        for i in qval_np[0]:
            s += np.exp(i/tau)
        for a in qval_np:
            p_a = np.exp(a/tau)
            prob = np.append(prob, (p_a/s))
            r = random.uniform(0,1)
            if r < prob[0]:
                return 0
            else:
                return 1


    def learn(self):
        self.count_N += 1
        minibatch = self.memory.get_mini_batch(self.batch_size)
        for interaction in minibatch:
            state = torch.Tensor(interaction[0]).reshape(1, 4) # fixme
            state_next = torch.Tensor(interaction[2]).reshape(1, 4) #fixme
            qvalues = self.qlearning_nn(state)
            action = interaction[1]
            reward = interaction[3]
            qval_prec = qvalues[0][action]
            if interaction[4]:
                tmp = torch.Tensor([reward]).reshape(1)
            else:
                qvalues_next = self.target_network(state_next)
                qmax = torch.max(qvalues_next[0])
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