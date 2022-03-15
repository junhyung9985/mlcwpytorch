import torch
import torch.nn as nn

class BPMLL(nn.Module):
    def __init__(self, input_size, output_size):
        super(BPMLL, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, input_size//2),
            nn.ReLU(),
            nn.Linear(input_size//2, output_size),
            nn.Sigmoid()
        )
    
    def forward(X):
        return self.network(X)

# To be implemented : BP-MLL Loss
    # E = 1 / |Y||Y_bar| sum_(k,l) in Y x Y_bar(exp(-ck-ci))
    # add up all E for each data instance
