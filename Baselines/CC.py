import torch
import torch.nn as nn
from torch.autograd import Variable

class CC(nn.Module):
    def __init__(self, input_size, output_size):
        super(CC,self).__init__()
        self.output_size = output_size
        self.network = nn.ModuleList()

        for i in range(output_size):
            self.network[i] = nn.Sequential(
                nn.Linear(input_size+i, 1),
                nn.Sigmoid()
            )

    def forward(self, X, y):
        Z = X
        out = []
        for i in range(self.output_size):
            z = self.network[i](Z)
            #z = (z > Variable(torch.Tensor([0.5]).to(self.device))).double() * 1
            out.append(z)
            Z = torch.cat((Z, y[i]), axis= 1)
        return torch.cat(out, axis = 1)

    def predict_proba(self, X):
        Z = X
        out = []
        for i in range(self.output_size):
            z = self.network[i](Z)
            out.append(z)
            z = (z > Variable(torch.Tensor([0.5]).to(self.device))).double() * 1
            Z = torch.cat((Z, z), axis= 1)
        return torch.cat(out, axis = 1)