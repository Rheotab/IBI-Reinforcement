import torch
import torch.nn.functional as F
import copy


class Net(torch.nn.Module):
    def __init__(self, D_out=4):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = torch.nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.nn1 = torch.nn.Linear(49 * 64, 512)
        self.nn2 = torch.nn.Linear(512, D_out)
        self.D_out = D_out


    def forward(self, x):
        x_conv1 = F.relu(self.conv1(x))
        # x = F.max_pool2d(x, 2)
        x_conv2 = F.relu(self.conv2(x_conv1))
        x_conv3 = F.relu(self.conv3(x_conv2))
        #x_pool = F.max_pool2d(x_conv2, 2)
        #x_v = F.relu(self.nn1(x_conv2.view(x_conv2.size(0), -1)))
        # x_flat = x_v.flatten()
        # x = x.view(self.nn1.shape())
        x_nn1 = F.relu(self.nn1(x_conv3.view(x_conv3.size(0), -1)))
        x_nn2 = F.relu(self.nn2(x_nn1))
        return x_nn2
