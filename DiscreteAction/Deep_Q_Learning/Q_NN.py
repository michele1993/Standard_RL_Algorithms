import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Q_net(nn.Module):

    def __init__(self, Input_size = 4, W1_size = 56, Output_size=2): #256 #56

        super(Q_net,self).__init__()

        self.ll_1 = nn.Linear(Input_size, W1_size)
        self.ll_2 = nn.Linear( W1_size, Output_size)



    def forward(self, x):

        x = F.relu(self.ll_1(x))
        x = self.ll_2(x)

        return x.view((-1,2))


    def disable_gradient(self):

        # why does this work?
        for parm_1, parm_2 in zip(self.ll_1.parameters(),self.ll_2.parameters()):

            parm_1.requires_grad_(False)
            parm_2.requires_grad_(False)

