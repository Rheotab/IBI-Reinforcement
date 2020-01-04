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
        self.qlearning_nn = DQN.DQN_two(32,32)
        self.target_network = DQN.DQN_two(32,32)
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
        self.arr_max_q_val_z = {
            "arr": [],
            "step": []
        }
        self.arr_max_q_val_o = {
            "arr": [],
            "step": []
        }
        self.step = 0

    def act(self, observation, reward, done):
        self.step += 1
        qvalues = self.qlearning_nn(torch.Tensor(observation).reshape(1, 4))

        #value = self.politique_greedy(qvalues)
        value = self.politique_boltzmann(qvalues, 0.5)
        return value

    def set_epsilon(self):
        self.eps = self.epsilon * (self.total_episode - self.episode) / self.total_episode

    def memorise(self, interaction):
        if interaction[4]:
            self.set_epsilon()
        self.memory.add(interaction)

    def set_ep(self):
        self.episode += 1

    def politique_greedy(self, qval):
        qval_np = qval.clone().detach().numpy()
        if qval_np[0][0] == qval_np[0][1]:
            a = random.randint(0, len(qval_np[0]) - 1)
        else:
            a = np.argmax(qval_np[0])
        if int(a) == 0:
            self.arr_max_q_val_z["arr"].append(qval_np[0][a])
            self.arr_max_q_val_z["step"].append(self.step)
        if int(a) == 1:
            self.arr_max_q_val_o["arr"].append(qval_np[0][a])
            self.arr_max_q_val_o["step"].append(self.step)
        if random.random() < self.eps:
            a = random.randint(0, len(qval_np[0]) - 1)
        return a

    def politique_boltzmann(self, qval, tau):
        qval_np = qval.clone().detach().numpy()
        s = 0
        #print(qval_np)
        prob = np.array([])
        for i in qval_np[0]:
            if i > 0:
                s += np.exp(i / tau)
        for a in qval_np[0]:
            if a > 0:
                p_a = np.exp(a / tau)
            else:
                p_a = 0
            prob = np.append(prob, (p_a / s))

        r = random.uniform(0, 1)
        sm = 0
        for j in range (len(prob)):
            sm += prob[j]
            if r < sm:
                return j

    def learn(self):
        minibatch = self.memory.get_mini_batch(self.batch_size)
        for interaction in minibatch:
            self.count_N += 1
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

    def learn_m(self):
        states, actions, next, rewards, done = self.memory.get_mini_batch_dim(self.batch_size)
        self.count_N += 1
        qvalues = self.qlearning_nn(states)
        qval_prec = []
        for i in range(self.batch_size):
            qval_prec.append(qvalues[i][actions[i]])
        qval_prec = torch.Tensor(qval_prec)
        qvalues_next = self.target_network(next)
        qmax = torch.max(qvalues_next, dim=1)
        y = done * (self.gamma * qmax.values) + rewards
        # loss = (qval_prec - y)**2
        loss = F.mse_loss(qval_prec, y)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        if self.N <= self.count_N:
            self.count_N = 0
            print("TARGET")
            self.target_network.load_state_dict(self.qlearning_nn.state_dict())

    def show_mean_loss_ep(self):
        plt.plot(self.arr_mean_loss)
        plt.title("Ep_Avg_LOSS")
        plt.show()

    def show_loss_learn(self):
        plt.plot(self.arr_loss)
        plt.title("LOSS")
        plt.ylabel("loss")
        plt.xlabel("step")
        plt.show()

    def show_max_val(self):
        plt.scatter(self.arr_max_q_val_o['step'], self.arr_max_q_val_o['arr'], s=0.5,label='Scatter Ones', color='r')
        plt.scatter(self.arr_max_q_val_z['step'],self.arr_max_q_val_z['arr'], s=0.5,label='Scatter Zeros', color='blue')
        plt.xlabel('Iteration')
        plt.ylabel('max Q value')
        plt.title('Q value progress')
        plt.legend()
        plt.show()