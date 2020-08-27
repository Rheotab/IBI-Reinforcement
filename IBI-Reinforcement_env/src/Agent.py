import torch
import torch.nn.functional as F
import torch.optim.rmsprop as rmsprop
import random
from DQN_Breakout import Net
from Memory import Memory
import numpy as np

from Tracker import Tracker


class Agent(object):
    def __init__(self, gamma, lr, buffer_size, update_target, epsilon, action_space, batch_size, pretrained_path=None):

        # CUDA variables
        self.USE_CUDA = torch.cuda.is_available()
        self.dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

        if pretrained_path is not None:
            # Load network from existing file.
            self.net = Net()
            self.net.load_state_dict(pretrained_path)
            self.target = Net()
            self.target.load_state_dict(self.net.state_dict())
            pass
        else:
            self.net = Net()
            self.target = Net()
            self.target.load_state_dict(self.net.state_dict())

        self.optimiser = rmsprop.RMSprop(self.net.parameters(), lr=lr, eps=epsilon)

        self.epsilon = epsilon  # Politic (exploration VS exploitation)
        self.gamma = gamma  # Discount of future rewards
        self.learning_rate = lr  # Learning rate
        self.buffer_size = buffer_size  # Size of memory (FIFO)
        self.update_target = update_target  # Iteration of learning process before update target.
        self.batch_size = batch_size

        self.action_space = action_space

        self.step = 0  # Number of action agent did.
        self.epoch = 0  # Number of forward/backward pass in convnet
        self.update_counter = 0  # Number of forward/backward between last reset (used for update target)
        self.memory = Memory(self.buffer_size)

        self.tracker = Tracker()

    def get_action(self, observation):
        self.step += 1
        qvalues = self.target(torch.Tensor(observation.reshape((1, 4, 84, 84)))) #FIXME
        value = int(self.politique_greedy(qvalues))
        if value == 0:
            print('NO OP ?????')
        self.tracker.add_act(value)
        self.tracker.add_qvalues(qvalues)
        return value

    def politique_greedy(self, qval):
        qval_np = qval.clone().detach().numpy()
        if random.random() < self.epsilon:
            v = int(self.action_space.sample())
            return v
        a = np.argwhere(qval_np[0] == np.amax(qval_np[0])).flatten()
        k = np.random.choice(a)
        return k

    def memorise(self, interaction):
        self.memory.add(interaction)


    def learn(self):
        states, actions, next, rewards, notdone = self.memory.get_mini_batch_dim(self.batch_size)
        self.epoch += self.batch_size
        qvalues = self.net(states)
        qval_prec = []
        for i in range(self.batch_size):
            qval_prec.append(qvalues[i][actions[i]])
        qval_prec = torch.tensor(qval_prec)  # FIXME
        qvalues_next = self.target(next)
        qmax = torch.max(qvalues_next, dim=1)
        y = notdone * (self.gamma * qmax.values) + rewards
        # loss = (qval_prec - y)**2
        loss = F.mse_loss(qval_prec, y, reduction='mean')
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()
        if self.update_counter == self.update_target:
            self.update_counter = 0
            self.target.load_state_dict(self.net.state_dict())

