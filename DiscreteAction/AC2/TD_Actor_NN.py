import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as opt




class Actor_net(nn.Module):

    def __init__(self,ln_rate,discount, Input_size=4, Hidden_size=20, output_size=2):

        super().__init__()

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size,output_size)


        self.discount = discount
        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x = torch.sigmoid(self.l2(x))

        return x

    def REINFORCE(self, p_action, advantage): # , done

        Policy_cost = - p_action * advantage.detach() # need minus because need to perform gradient ascent

        self.optimiser.zero_grad()
        Policy_cost.backward()
        self.optimiser.step()

        return Policy_cost