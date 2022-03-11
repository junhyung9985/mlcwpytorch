import torch
import torch.nn as nn

class STA(nn.Module):
    def __init__(self, input_size, output_size):
        super(STA,self).__init__()
        self.network1 = nn.Linear(input_size, output_size)
        self.network2 = nn.Linear(input_size+output_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, train_second_layer = False):
        out = self.network1(X)
        out = self.sigmoid(out)
        if train_second_layer:
            X = torch.cat((X, out), axis= 1)
            out = self.network2(X)
            out = self.sigmoid(out)
            
        return out