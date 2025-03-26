import torch.nn as nn
import torch


class CNNQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNQNetwork, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_dim, 32, kernel_size=8, stride=4),  # 32 filters, 8x8 kernel
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )


        with torch.no_grad():
            self.feature_size = self.conv(torch.zeros(1, input_dim, 84, 84)).shape[1]

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  
        )

    def forward(self, x):
        x = self.conv(x)
        return self.fc(x)