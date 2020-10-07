import torch
import torch.nn.functional as F
import copy
import torch.nn as nn

'''
Dueling DQN
Value and Advantage
'''


class Net(torch.nn.Module):
    def __init__(self, D_out=4):
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        self.fc_input_dim = 49 * 64
        self.value = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage = nn.Sequential(
            nn.Linear(self.fc_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, D_out)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.view(features.size(0), -1)
        values = self.value(features)
        advantages = self.advantage(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
