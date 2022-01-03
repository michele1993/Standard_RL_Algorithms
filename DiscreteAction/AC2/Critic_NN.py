import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as opt

class Critic_NN(nn.Module):

    def __init__(self,ln_rate,discount, n_inputs = 4,n_hiddens = 20, n_outputs=1):

        super().__init__()
        self.discount= discount

        self.l1 = nn.Linear(n_inputs,n_hiddens)
        self.l2 = nn.Linear(n_hiddens, n_outputs)

        self.I = 1
        self.optimiser = opt.Adam(self.parameters(), ln_rate)

    def forward(self, x):

        x = F.relu(self.l1(x))
        x= self.l2(x)

        return x

    def advantage(self,c_state,n_state,rwd, done):

        vs_1 = self(torch.tensor(c_state))

        vs_2 = self(torch.tensor(n_state)).detach()

        td_error = (rwd + (1 - done) * self.discount * vs_2  - vs_1)

        return td_error

    def update(self, TD_error):

        cost = TD_error**2
        self.optimiser.zero_grad()
        cost.backward()
        self.optimiser.step()



