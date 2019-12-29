import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
import random
from gym import wrappers, logger
from DQN import DQN
from Memory import Memory
import numpy as np
import time
import copy

class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, action_space, buffer_size=10000, epsilon=0.3, batch_size=10, gamma=0.05, eta=0.001, N = 100):
        self.action_space = action_space
        self.eps = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.eta = eta
        self.N = N
        self.count_N = 0
        self.memory = Memory(buffer_size)
        self.qlearning_nn = DQN(64)
        self.target_network = DQN(64)
        self.target_network.load_state_dict(self.qlearning_nn.state_dict())
        self.optimiser = torch.optim.Adam(self.qlearning_nn.parameters(), lr=self.eta)

        self.arr_loss = []



    def act(self, observation, reward, done):
        qvalues = self.qlearning_nn(torch.Tensor(observation).reshape(1,4))
        #qvalues = self.target_network(torch.Tensor(observation).reshape(1, 4))
        #self.politique_boltzmann(qvalues, 0.1)
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
        #print(len(minibatch))
        start = time.time()
        for interaction in minibatch:
            state = torch.Tensor(interaction[0]).reshape(1, 4)
            state_next = torch.Tensor(interaction[2]).reshape(1, 4)
            qvalues = self.qlearning_nn(state)
            #etat_prec = torch.Tensor(interaction[0])
            action = interaction[1]
            reward = interaction[3]
            # etat_suiv = torch.Tensor(interaction[2])
            qval_prec = qvalues[0][action]
            if interaction[4]:
                # loss = torch.from_numpy(np.array([(qval_prec - reward)**2]))
                #loss = torch.Tensor([float((qval_prec - reward)**2)])
                tmp = torch.Tensor([reward]).reshape(1)
            else:
                #qvalues_next = self.qlearning_nn(torch.Tensor(interaction[2].reshape(1,4)))
                qvalues_next = self.target_network(state_next)
                qmax = torch.max(qvalues_next[0])
                #print(qval_prec.shape)

                tmp = torch.Tensor([reward + self.gamma * qmax]).reshape(1)
                # loss = torch.from_numpy(np.array([(qval_prec - (reward + gamma * qmax))**2]))
               # loss = torch.Tensor([float((qval_prec - (reward + gamma * qmax))**2)])
            loss = F.mse_loss(qval_prec.reshape(1), tmp)
            self.arr_loss.append(loss)
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        if self.N == self.count_N:
            self.count_N = 0
           # self.target_network = copy.deepcopy(self.qlearning_nn)
            self.target_network.load_state_dict(self.qlearning_nn.state_dict())


    def show_loss(self):
        plt.plot(self.arr_loss)
        plt.show()

        #print(time.time() - start)
