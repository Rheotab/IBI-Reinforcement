import torch
import torch.nn.functional as F
import torch.optim.rmsprop as rmsprop
import torch.nn as nn
import random
from DQN_Breakout import Net
from Memory import Memory
import numpy as np



class Agent(object):
    def __init__(self, env, gamma=0.99, lr=0.001, buffer_size=100000, epsilon=0.20, batch_size=32):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Net().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()
        self.epsilon = epsilon  # Politic (exploration VS exploitation)
        self.gamma = gamma  # Discount of future rewards
        self.learning_rate = lr  # Learning rate
        self.buffer_size = buffer_size  # Size of memory (FIFO)
        self.batch_size = batch_size
        self.step = 0  # Number of action agent did.
        self.nb_element_learned = 0  # Number of forward/backward pass in convnet
        self.memory = Memory(self.buffer_size)
        self.action_space = env.action_space

    def get_action(self, observation):
        self.step += 1
        observation = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        # observation.reshape((1, 4, 84, 84)
        qvalues = self.model.forward(observation) #FIXME
        return int(self.politique_greedy(qvalues))

    def politique_greedy(self, qval):
        if random.random() > self.epsilon:
            v = int(self.action_space.sample())
            return v
        return np.argmax(qval.cpu().detach().numpy())

    def memorise(self, interaction):
        self.memory.add(interaction)

    def compute(self, batch):
        states, actions, next, rewards, dones = batch
        self.nb_element_learned += self.batch_size

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next = torch.FloatTensor(next).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        current_Q = current_Q.squeeze(1)
        next_Q = self.model.forward(next)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards + self.gamma * max_next_Q

        loss = self.MSE_loss(current_Q, expected_Q)

        return loss

    def update(self):
        batch = self.memory.get_mini_batch_dim(self.batch_size)
        loss = self.compute(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
