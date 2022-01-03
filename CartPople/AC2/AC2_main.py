import torch
import gym

from torch.distributions import Categorical

from AC2.TD_Actor_NN import Actor_net
from AC2.Critic_NN import Critic_NN

torch.manual_seed(958)
env = gym.make("CartPole-v0")

n_episodes = 2000
max_t_steps = 200
discount= 0.99
ln_rate_c = 0.005
ln_rate_a = 0.001
pre_train = 250


actor = Actor_net(ln_rate_a,discount = discount).double()
critic = Critic_NN(ln_rate_c,discount = discount).double()




av_return = []
training_acc = []

for ep in range(n_episodes):

    c_state = env.reset()

    t = 0
    done = False

    while not done:

        mean_action = actor(torch.tensor(c_state))

        d = Categorical(mean_action)

        action = d.sample()

        lp_action = d.log_prob(action)

        n_state,rwd,done,_ = env.step(action.numpy())

        TD_error = critic.advantage(c_state,n_state,rwd,done)
        critic.update(TD_error)

        # Start updating Actor after a small critic pre-training session
        if ep > pre_train:
            rf_cost = actor.REINFORCE(lp_action,TD_error)

        c_state = n_state
        t+=1


    av_return.append(t)

    if ep %100 == 0:

        avr_acc = sum(av_return)/100
        print("ep: ",ep," av_return: ", avr_acc)

        av_return = []
        training_acc.append(avr_acc)


# Save training accuracy
#torch.save(training_acc,"/Users/michelegaribbo/Desktop/Results/AC2_accuracy.pt")






