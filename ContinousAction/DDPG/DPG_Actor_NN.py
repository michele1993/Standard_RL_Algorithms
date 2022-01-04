import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt


class Actor_NN(nn.Module):

    def __init__(self, Input_size=3, Hidden_size=56, Output_size=1,max_action=2,ln_rate = 1e-3):

        super().__init__()
        self.max_action = max_action

        self.l1 = nn.Linear(Input_size, Hidden_size)
        self.l2 = nn.Linear(Hidden_size, Hidden_size)
        self.l3 = nn.Linear(Hidden_size, Output_size)
        self.optimiser = opt.Adam(self.parameters(),ln_rate)


    def forward(self, x):

        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.max_action

        return x


    def freeze_params(self):

        for params in self.parameters():

            params.requires_grad = False

    def update(self,loss):

        loss = torch.mean(loss)
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def copy_weights(self, estimate):

        for t_param, e_param in zip(self.parameters(), estimate.parameters()):
            t_param.data.copy_(e_param.data)

    def soft_update(self, estimate, decay):

        with torch.no_grad():

            # do polyak averaging to update target NN weights
            for t_param, e_param in zip(self.parameters(),estimate.parameters()):
                t_param.data.copy_(t_param.data * decay + (1 - decay) * e_param.data)
