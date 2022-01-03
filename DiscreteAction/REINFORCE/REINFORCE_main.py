
import torch
import gym
import torch.optim as opt
import numpy as np


torch.manual_seed(958)

from torch.distributions import Categorical

from REINFORCE.Policy_network import Policy_net
from REINFORCE.Baseline_net import Baseline_nn

n_episodes= 2000
discount = 0.99
learning_rate = 1e-3
batch_size = 1

env = gym.make("CartPole-v0")
pol_nn = Policy_net(output_size =2)#.double()
base_nn = Baseline_nn()#.double()

criterion = torch.nn.MSELoss()

params = list(pol_nn.parameters()) + list(base_nn.parameters())

optimiser = opt.Adam(params,learning_rate)

episode_overall_return = []
training_acc = []


for i in range(n_episodes):

    current_st = env.reset()
    episode_rwd = []
    episode_lp_action = []
    episode_states = torch.empty(0)

    t = 0
    done = False

    while not done:

        current_st = torch.FloatTensor([current_st])

        episode_states = torch.cat([episode_states,current_st])


        mean_action = pol_nn(torch.tensor(current_st))

        d = Categorical(mean_action) # try to replace with bernulli and single output

        action = d.sample()

        episode_lp_action.append(d.log_prob(action))

        next_st, rwd, done, _ = env.step(int(action.numpy()))

        episode_rwd.append(torch.FloatTensor([rwd]))

        current_st = next_st

        t +=1





    predicted_value = base_nn(episode_states.view(-1,4))


    episode_rwd = torch.cat(episode_rwd)

    episode_lp_action = torch.cat(episode_lp_action)

    episode_rwd = pol_nn.compute_returns(episode_rwd,discount)


    advantage = episode_rwd.view(-1) - predicted_value.view(-1) # v_value



    # Update Actor:

    policy_c = sum(pol_nn.REINFORCE(episode_lp_action,advantage))

    baseline_c = sum(torch.pow(advantage, 2))

    loss =  policy_c + baseline_c  #pol_nn.REINFORCE(episode_lp_action[e],advantage) + torch.pow(episode_rwd[e] - episode_v_value[e], 2)

    optimiser.zero_grad()

    loss.backward()

    optimiser.step()

    episode_overall_return.append(t)


    if i % 100 == 0:

        aver_acc = sum(episode_overall_return)/100
        print("Baseline loss {}, Policy cost {}, Return {}, Episode {}".format(baseline_c.data/t,policy_c.data/t,aver_acc, i))

        episode_overall_return = []
        training_acc.append(aver_acc)

# Save training accuracy
torch.save(training_acc,"/Users/michelegaribbo/Desktop/Results/REINFORCE_accuracy.pt")
