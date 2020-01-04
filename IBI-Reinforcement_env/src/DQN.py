import torch
import torch.nn.functional as F
import copy


class DQN_one(torch.nn.Module):
    def __init__(self, D_H, D_out=2):
        super(DQN_one, self).__init__()
        self.nn1 = torch.nn.Linear(4, D_H)
        self.nn2 = torch.nn.Linear(D_H, D_out)
        torch.nn.init.xavier_uniform_(self.nn1.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(self.nn2.weight, gain=torch.nn.init.calculate_gain('relu'))
        #torch.nn.init.uniform_(self.nn2.weight)
        #torch.nn.init.uniform_(self.nn1.weight)
        self.D_H1 = D_H
        self.D_out = D_out

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        return x


class DQN_two(torch.nn.Module):
    def __init__(self, D_H1, D_H2, D_out=2):
        super(DQN_two, self).__init__()
        self.nn1 = torch.nn.Linear(4, D_H1)
        self.nn2 = torch.nn.Linear(D_H1, D_H2)
        self.nn3 = torch.nn.Linear(D_H2, D_out)
        #torch.nn.init.xavier_normal_(self.nn1.weight, gain=torch.nn.init.calculate_gain('relu'))
        #torch.nn.init.xavier_normal_(self.nn2.weight, gain=torch.nn.init.calculate_gain('relu'))
        #torch.nn.init.xavier_normal_(self.nn3.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.normal_(self.nn1.weight)
        torch.nn.init.normal_(self.nn2.weight)
        torch.nn.init.normal_(self.nn3.weight)
        self.D_H1 = D_H1
        self.D_H2 = D_H2
        self.D_out = D_out

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        x = F.relu(self.nn3(x))
        return x


class DQN_three(torch.nn.Module):
    def __init__(self, D_H1, D_H2, D_H3, D_out=2):
        super(DQN_three, self).__init__()
        self.nn1 = torch.nn.Linear(4, D_H1)
        self.nn2 = torch.nn.Linear(D_H1, D_H2)
        self.nn3 = torch.nn.Linear(D_H2, D_H3)
        self.nn4 = torch.nn.Linear(D_H3, D_out)
        self.D_H1 = D_H1
        self.D_H2 = D_H2
        self.D_H3 = D_H3
        self.D_out = D_out

    def forward(self, x):
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        x = F.relu(self.nn3(x))
        x = F.relu(self.nn4(x))
        return x
