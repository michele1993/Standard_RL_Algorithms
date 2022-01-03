import torch
import torch.nn as nn

class Baseline_nn(nn.Module):

    def __init__(self, input_size=4, output_size=1):

        super(Baseline_nn,self).__init__()

        self.l1 = nn.Linear(input_size,output_size)

    def forward(self,x):

        x = self.l1(x)

        return x #.view(-1,1)

