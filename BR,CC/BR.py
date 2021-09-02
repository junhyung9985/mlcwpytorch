import torch
import torch.nn as nn

class BR(nn.Module):
    def __init__(self, input_size, output_size):
        super(BR,self).__init__()
        self.network = nn.Linear(input_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):
        out = self.network(X)
        out = self.sigmoid(out)
        return out