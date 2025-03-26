import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from DQNetwork import *
from Memory import Memory

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon=1.0, epsilon_decay=0.9995, epsilon_min=0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.gamma = gamma

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Use simple CNN-based DQN
        self.q_network = CNNQNetwork(state_dim, action_dim).to(self.device)
        self.target_network = CNNQNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = Memory(10000)
        self.next_q_values_save = []
        self.loss_save = []

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            with torch.no_grad():
                return self.q_network(state).argmax().item()

    def add_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)

    def train(self, batch_size=32):
        if self.replay_buffer.size() < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert lists to numpy arrays first (Optimized)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32, device=self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        s_shape = states.shape
        ns_shape = next_states.shape

        # Compute Q-values
        q_values = self.q_network(states).gather(1, actions).squeeze()

        # Compute Double DQN target Q-values
        next_actions = self.q_network(next_states).argmax(1, keepdim=True)
        next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        

        # Hubber loss vs MSE
        loss = nn.MSELoss()(q_values, target_q_values.detach())

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient Clipping to prevent exploding gradients
     #   torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.next_q_values_save.append(next_q_values.detach().numpy())
        self.loss_save.append(loss.item())
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def print_avg_qvalue(self):
        print(np.mean(self.next_q_values_save))
        print(np.mean(self.loss_save))
        self.next_q_values_save = []
        self.loss_save  = []

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())
