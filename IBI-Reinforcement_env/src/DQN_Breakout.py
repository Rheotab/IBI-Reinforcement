import torch
import torch.nn.functional as F
import copy


class Net(torch.nn.Module):
    def __init__(self, D_H, D_out=2):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 20, kernel_size=1, stride=1)
        print(self.conv1.weight.shape)
        self.conv2 = torch.nn.Conv2d(20, 20, kernel_size=1, stride=1)
        self.nn1 = torch.nn.Linear(20, D_H)
        self.nn2 = torch.nn.Linear(D_H, D_out)
        self.D_H = D_H
        self.D_out = D_out


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        return x
