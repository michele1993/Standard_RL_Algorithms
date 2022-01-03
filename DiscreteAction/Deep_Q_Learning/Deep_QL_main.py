import gym
import torch.nn as nn
from Deep_Q_Learning.Q_NN import Q_net
import numpy as np
import torch
import random

env = gym.make("CartPole-v0")
Buffer_size = 2000
n_episodes = 2500
epsilon = 1
e = 0
discount = 0.95
learning_rate = 1e-4
batch_size = 60
epsilon_decay = 0.999
pre_training = 500

M_buffer = []
net_1 = Q_net()
net_2 = Q_net()

net_1 = net_1.double()
net_2.load_state_dict(net_1.state_dict())

net_2.double()

net_1.disable_gradient()

optimizer = torch.optim.Adam(net_2.parameters(), lr=learning_rate)
criterion = nn.MSELoss()
done = False

episode_len = []
training_acc = []



for i in range(0,n_episodes):


    current_state = env.reset()

    done = False

    t_2 = 0


    while not done:

        optimizer.zero_grad()

        random_value = np.random.rand()

        if random_value > epsilon:

            with torch.no_grad():
                _, action = torch.max(net_2(torch.from_numpy(current_state)),axis=1)


            action = int(action.numpy())


        else:

            action = env.action_space.sample()


        next_state, rwd, done, _ = env.step(action)


        t_2+=1

        # Store experience in the buffer:

        if e < Buffer_size:

            M_buffer.append((current_state,action,rwd,next_state,done))
            e += 1

        else:

            if e % Buffer_size == 0:
                e = 0

            M_buffer[e] = ((current_state,action,rwd,next_state,done))
            e+=1


        current_state = next_state


        # Start learning policy after some pre-training storing transition in the buffer
        if i > pre_training:

            transitions = random.sample(M_buffer,batch_size)

            b_sampled_c_state, b_sampled_a, b_sampled_rwd, b_sampled_n_state, b_sampled_done = zip(*transitions)

            b_sampled_n_state = torch.tensor(b_sampled_n_state)

            b_sampled_rwd = torch.tensor(b_sampled_rwd).double()

            b_sampled_done = list(b_sampled_done)


            with torch.no_grad():
                Target = b_sampled_rwd + discount * torch.max(net_1(b_sampled_n_state),axis=1)[0] # greedy target (i.e. off-policy), need [0] because torch.max returns index too

            Target[b_sampled_done] = b_sampled_rwd[b_sampled_done]


            input = torch.tensor(b_sampled_c_state)

            current_estimate = net_2(input) # from behavioural policy (i.e. e-greedy) net_2(torch.tensor([b_sampled_c_state])) - why does this give completely different backward graph?


            indexes = [torch.arange(0,len(current_estimate)),b_sampled_a]

            loss = criterion(Target,current_estimate[indexes])

            # Perform Update of net_2:
            loss.backward()
            optimizer.step()


    episode_len.append(t_2)

    if i%100 == 0:

        current_accuracy = sum(episode_len)/len(episode_len)
        print("\n Accuracy for last 100 trajectories: ",current_accuracy, "at overall iteration: ", i)

        training_acc.append(current_accuracy)

        episode_len = []

    epsilon*= epsilon_decay

     # When change weights for old network place it in no gradient block
    if i % 10 == 0:

        with torch.no_grad(): # probably not needed

            net_1.load_state_dict(net_2.state_dict())
            net_1.disable_gradient()


# Save training accuracy:
#torch.save("/Users/michelegaribbo/Desktop/Results/DeepQLearning_accuracy.pt",training_acc)







