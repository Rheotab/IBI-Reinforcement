import torch
import torch.nn.functional as F


class DQN(torch.nn.Module):
    def __init__(self, D_H, D_out=2):
        super(DQN, self).__init__()
        # self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=2) # stripe / padding
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=2)
        self.nn1 = torch.nn.Linear(4, D_H)
        self.nn2 = torch.nn.Linear(D_H, D_out)

    def forward(self, x):
        # x = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        # x = F.relu(self.conv2(x))
        # x = F.max_pool2d(x, 2)
        # x = torch.flatten(x, 1)
        x = F.relu(self.nn1(x))
        x = F.relu(self.nn2(x))
        return F.log_softmax(x, dim=1)
