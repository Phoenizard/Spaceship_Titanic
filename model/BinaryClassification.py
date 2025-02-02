from torch import nn
import torch

class BinaryClassification(nn.Module):
    def __init__(self, input_dim=12):
        super(BinaryClassification, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),                      
        )

    def forward(self, x):
        return self.net(x)