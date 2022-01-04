import gym
import torch
import numpy as np
from TD3.Actor_Critic import Actor_NN,Critic_NN
from TD3.TD3_alg import TD3
from TD3.Memory_Buffer import V_Memory_B

dev = torch.device('cpu')
#dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


#TD_3 parameters:
n_episodes = 10000
buffer_size = 100000
batch_size = 100 #  number of transition bataches (i.e. n_arms) sampled from buffer
start_update = 50
actor_update = 2
ln_rate_c = 0.001#0.00002
ln_rate_a = 0.001#0.00001
decay_upd = 0.005
std = 0.1

# Environment:
env = gym.make('BipedalWalker-v3')
action_space = env.action_space.shape[0]
state_space = env.observation_space.shape[0]
max_t_step = 2000
t_print = 10


#Initialise actor and critic
agent = Actor_NN(ln_rate = ln_rate_a).to(dev)

critic_1 = Critic_NN(ln_rate=ln_rate_c).to(dev)
critic_2 = Critic_NN(ln_rate=ln_rate_c).to(dev)

#Initialise Buffer:
buffer = V_Memory_B(dev,a_space= action_space,s_space = state_space, batch_size=batch_size,size=buffer_size)

# Initialise DPG algorithm passing all the objects
td3 = TD3(agent,critic_1,critic_2,buffer,decay_upd,dev, actor_update= actor_update)


tot_critc_loss = []
tot_actor_loss = []
tot_rwd = []
training_acc = []

for ep in range(n_episodes):

    c_state = torch.FloatTensor(env.reset()).to(dev)
    done = False
    ep_rwd = 0
    ep_critc_loss = []
    ep_actor_loss = []

    for t in range(max_t_step):

        det_action = agent(c_state).detach()
        rand_explor = torch.randn(action_space).to(dev) * std
        action = (det_action + rand_explor).clamp(-1,1)

        n_s,rwd,done,_ = env.step(action.cpu().numpy())

        n_s = torch.FloatTensor(n_s)

        buffer.store_transition(c_state,action,torch.FloatTensor(np.asarray(rwd,dtype=np.float)),n_s,torch.FloatTensor(np.asarray(done)))

        c_state = n_s

        if  ep > start_update:# and step % 10 == 0: #t%25 == 0 and

            critic_loss1,_,actor_loss = td3.update(t)
            ep_critc_loss.append(critic_loss1.detach())
            ep_actor_loss.append(actor_loss.detach())

        ep_rwd += rwd

        if done:
            break


    tot_rwd.append(ep_rwd)
    tot_critc_loss.append(sum(ep_critc_loss)/(t))
    tot_actor_loss.append(sum(ep_actor_loss) /(t))


    if ep % t_print == 0:

        aver_rwd = sum(tot_rwd)/t_print

        print("ep: ", ep)
        print("Aver final rwd: ", aver_rwd)
        print("Critic loss: ", sum(tot_critc_loss)/(t_print))
        print("Actor loss", sum(tot_actor_loss)*2 / (t_print))

        if aver_rwd >= 300: # should be averaged over past 100 ep
            print("Solved")
            break

        training_acc.append(aver_rwd)
        tot_rwd = []
        tot_critc_loss = []
        tot_actor_loss = []


# Save results
#torch.save(training_acc,"/Users/michelegaribbo/Desktop/Results/TD3_accuracy.pt")

