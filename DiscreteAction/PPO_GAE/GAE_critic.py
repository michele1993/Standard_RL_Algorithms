import torch.nn as nn
import torch.nn.functional as F
import torch

class Critic_NN(nn.Module):


    def __init__(self, discount,lambda_parm = 0.95, input_size = 3, n_hiddens = 56, n_outputs=1,ln_rate = 1e-3): #0.95

        super().__init__()
        self.gamma = discount
        self.lambda_p = lambda_parm

        self.l1 = nn.Linear(input_size,n_hiddens)
        self.l2 = nn.Linear(n_hiddens, n_outputs)



    def forward(self, x):

        x = F.relu(self.l1(x))
        x= self.l2(x)

        return x

    def GAE(self, values, last_v_val,rewards, masks):

        returns = []

        values = values + [last_v_val]

        gae = torch.tensor([0])

        # start from last t step and go backward, multiplying by lambda and gamma at each step the GAE, to eansure accurate weighting for each TD_t error
        for i in range(len(rewards)-1,-1,-1):

            delta = rewards[i] + self.gamma * values[i+1] * masks[i] - values[i] # values has one extra entry that rewards due to final state, need to values[i+1].detach() ?
            gae = delta + self.lambda_p * self.gamma * masks[i] * gae

            returns.insert(0,gae + values[i])

        return returns
