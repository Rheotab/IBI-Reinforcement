import torch
import torch.nn.functional as F
import copy


class Net(torch.nn.Module):
    def __init__(self, D_H, D_out=4):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=5, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=4)
        self.nn1 = torch.nn.Linear(1024, D_H)
        self.nn2 = torch.nn.Linear(D_H, D_out)
        self.D_H = D_H
        self.D_out = D_out


    def forward(self, x):
        x = F.relu(self.conv1(x))
        #x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        #x = F.max_pool2d(x, 2)
        x = x.flatten()
        # x = x.view(self.nn1.shape())
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        return x
