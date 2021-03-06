import torch.nn.functional as F
import torch.nn as nn


class Actor_NN(nn.Module):

    def __init__(self, input_size =3,h_size=56,ouput_size=1):

        super().__init__()

        self.l1= nn.Linear(input_size,h_size)
        self.l2 = nn.Linear(h_size,ouput_size)


    def forward(self,x):

        x = F.relu(self.l1(x))
        x = F.softmax(self.l2(x))


        return x
