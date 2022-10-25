from torch import nn
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class NARX(nn.Module):
    def __init__(self, n_features=10):
        super().__init__()
        self.lin = nn.Linear(n_features, 80)
        self.lin2 = nn.Linear(80, 80)
        self.lin3 = nn.Linear(80, 1)
        self.relu = nn.ReLU6()

    def forward(self, xb):
        z = self.lin(xb)
        z = self.relu(z)
        z = self.lin2(z)
        z = self.relu(z)
        z = self.lin3(z)
        return z