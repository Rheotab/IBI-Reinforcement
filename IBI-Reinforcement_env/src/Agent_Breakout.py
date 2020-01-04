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
    def __init__(self, action_space, buffer_size=100000, epsilon=0.1, batch_size=50, gamma=0.8, eta=0.005,
                 N=100):
        self.action_space = action_space
        self.eps = epsilon  # e-greedy
        self.batch_size = batch_size  # what do we learn
        self.gamma = gamma  # how much is the importance of reward in learning
        self.eta = eta  # learning rate
        self.N = N  # When do we transfer to target -> Improve stability
        self.count_N = 0
        self.memory = Memory(buffer_size)
        self.qlearning_nn = Net()
        self.target_network = Net()
        self.target_network.load_state_dict(self.qlearning_nn.state_dict())
        #self.optimiser = torch.optim.RMSprop(self.qlearning_nn.parameters(), lr=self.eta, momentum=0.95, eps=1e-2)
        self.optimiser = torch.optim.Adam(self.qlearning_nn.parameters(), lr=eta)
        self.arr_loss = []
        self.count_no_op = 0
        self.no_op_max = 30
        self.arr_q_value = []
        for i in range(4):
            self.arr_q_value.append({
                "step": [],
                "Q": []
            })
        self.step = 0

    def act(self, observation, reward, done):
        self.step += 1
        qvalues = self.qlearning_nn(torch.Tensor(observation.reshape((1,4,84,84))))
        value = int(self.politique_greedy(qvalues))
        if value == 0:
            self.count_no_op += 1
        return value

    def reset_noop(self):
        self.count_no_op = 0

    def memorise(self, interaction):
        self.memory.add(interaction)

    def politique_greedy(self, qval):
        qval_np = qval.clone().detach().numpy()
        # print(qval_np)
        index = int(np.argmax(qval_np))
        self.arr_q_value[index]["step"].append(self.step)
        self.arr_q_value[index]["Q"].append(np.max(qval_np))
        if random.random() < self.eps:
            v = int(self.action_space.sample())
            while self.count_no_op > self.no_op_max and v == 0:
                v = int(self.action_space.sample())
            return v
        a = np.argwhere(qval_np[0] == np.amax(qval_np[0])).flatten()
        k = np.random.choice(a)
        if self.count_no_op > self.no_op_max and int(k) == 0:
            a = np.argwhere(qval_np[0][1:] == np.amax(qval_np[0][1:])).flatten()
            k = np.random.choice(a) + 1
        return k



    def random_act(self):
        return int(self.action_space.sample())

    # FIXME index
    def politique_boltzmann(self, qval, tau):
        # TODO
        pass

    def learn(self):
        minibatch = self.memory.get_mini_batch(self.batch_size)
        for interaction in minibatch:
            self.count_N += 1
            state = torch.Tensor(interaction[0])
            state_next = torch.Tensor(interaction[2])
            qvalues = self.qlearning_nn(state)
            action = interaction[1]
            reward = interaction[3]
            qval_prec = qvalues[0][action]
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
                print("TARGET")
                self.target_network.load_state_dict(self.qlearning_nn.state_dict())

    def learn_m(self):
        states, actions, next, rewards, done = self.memory.get_mini_batch_dim(self.batch_size)
        self.count_N += self.batch_size
        qvalues = self.qlearning_nn(states)
        qval_prec = []
        for i in range(self.batch_size):
            qval_prec.append(qvalues[i][actions[i]])
        qval_prec = torch.Tensor(qval_prec)
        qvalues_next = self.target_network(next)
        qmax = torch.max(qvalues_next, dim=1)
        y = done * (self.gamma * qmax.values) + rewards
        # loss = (qval_prec - y)**2
        loss = F.mse_loss(qval_prec, y, reduction='mean')
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        if self.N <= self.count_N:
            self.count_N = 0
            print("TARGET")
            self.target_network.load_state_dict(self.qlearning_nn.state_dict())

    def show_loss(self):
        plt.plot(self.arr_loss)
        plt.xlabel("Learn step")
        plt.ylabel("LOSS")
        plt.show()

    def show_q_values(self):
        colors = ['red', 'blue', 'yellow', 'green']
        for i in range(4):
            plt.scatter(self.arr_q_value[i]["step"],self.arr_q_value[i]["Q"],s=0.5, label="Action : " + str(i),
                        color=colors[i])
        plt.xlabel('step')
        plt.ylabel('max Q value')
        plt.title('Q value progress')
        plt.legend()
        plt.show()

    def how_many_did_u_see(self):
        return str(self.memory.count)

    def save_model(self, path='/tmp/model_breakout'):
        torch.save(self.qlearning_nn.state_dict(), path + '.pth')
