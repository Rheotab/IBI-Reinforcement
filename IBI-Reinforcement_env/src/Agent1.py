import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import gym
import random
from gym import wrappers, logger
import DQN
from Memory import Memory
import numpy as np
import time
import copy


class Agent(object):
    def __init__(self, nb_ep, action_space, buffer_size, epsilon, batch_size, gamma, eta, N):
        self.action_space = action_space
        self.eps = epsilon
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = gamma
        self.eta = eta
        self.N = N
        self.count_N = 0
        self.memory = Memory(buffer_size)
        self.qlearning_nn = DQN.DQN_two(128,256)
        self.target_network = DQN.DQN_two(128,256)
        self.target_network.load_state_dict(self.qlearning_nn.state_dict())
        self.optimiser = torch.optim.Adam(self.qlearning_nn.parameters(), lr=self.eta)
        self.episode = 0
        self.total_episode = nb_ep
        self.arr_loss = []
        self.arr_mean_loss = []
        self.ep_current_loss = {
            "count": 0,
            "value": 0
        }

    def act(self, observation, reward, done):
        qvalues = self.qlearning_nn(torch.Tensor(observation).reshape(1, 4))
        value = self.politique_greedy(qvalues)
        return value

    def set_epsilon(self):
        self.eps = self.epsilon * (self.total_episode - self.episode) / self.total_episode

    def memorise(self, interaction):
        if interaction[4]:
            self.episode += 1
            self.set_epsilon()
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
            s += np.exp(i / tau)
        for a in qval_np:
            p_a = np.exp(a / tau)
            prob = np.append(prob, (p_a / s))
            r = random.uniform(0, 1)
            if r < prob[0]:
                return 0
            else:
                return 1

    def learn(self):
        minibatch = self.memory.get_mini_batch(self.batch_size)
        start = time.time()
        self.count_N += 1
        for interaction in minibatch:
            state = torch.Tensor(interaction[0]).reshape(1, 4)
            state_next = torch.Tensor(interaction[2]).reshape(1, 4)
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
            if interaction[4]:
                if self.ep_current_loss["count"] != 0:
                    self.arr_mean_loss.append(self.ep_current_loss["value"] / self.ep_current_loss["count"])
                self.ep_current_loss["value"] = 0
                self.ep_current_loss["count"] = 0
            else:
                self.ep_current_loss["value"] += loss.item()
                self.ep_current_loss["count"] += 1
            self.optimiser.zero_grad()
            loss.backward()
            self.optimiser.step()
        if self.N == self.count_N:
            self.count_N = 0
            print("TARGET")
            # self.target_network = copy.deepcopy(self.qlearning_nn)
            self.target_network.load_state_dict(self.qlearning_nn.state_dict())

    def show_mean_loss_ep(self):
        plt.plot(self.arr_mean_loss)
        plt.title("Ep_Avg_LOSS")
        plt.show()

    def show_loss_learn(self):
        plt.plot(self.arr_loss)
        plt.title("LOSS")
        plt.show()

