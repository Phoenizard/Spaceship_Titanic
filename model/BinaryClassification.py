from torch import nn
import torch

class BinaryClassification(nn.Module):
    def __init__(self, input_dim=12):
        super(BinaryClassification, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)