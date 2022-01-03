import gym
import torch
import torch.nn
from torch.distributions import Categorical

from Policy_gradient.PPO_GAE.Actor_NN import Actor_NN
from Policy_gradient.PPO_GAE.GAE_critic import Critic_NN
from Policy_gradient.PPO_GAE.PPO_alg import PPO





n_episodes= 2000
PPO_steps = 200
std = 0.3
gamma = 0.99
ln_rate_c = 0.001
ln_rate_a = 0.0001
t_printing = 100

torch.manual_seed(958)
env = gym.make("CartPole-v1")




model_actor = Actor_NN(input_size =env.observation_space.shape[0], ouput_size= 2)#.double() #env.action_space.shape[0]
model_critc = Critic_NN(discount= gamma,input_size =env.observation_space.shape[0])#.double()

ppo = PPO(model_actor,model_critc)



accuracy = []
training_acc = []

actor_cost = 0
critic_cost = 0

for ep in range(n_episodes):

    states = []

    actions = []

    values = []

    rewards = []

    log_ps = []

    masks = []


    c_state = env.reset()

    itr = 0

    for itr in range(PPO_steps):

        c_state = torch.FloatTensor(c_state)
        action_mean = model_actor(c_state)
        v_value = model_critc(c_state)


        d = Categorical(action_mean)
        action = d.sample()
        log_p = d.log_prob(action)


        obs, rwd, done, _ = env.step(action.numpy())
        mask = not done

        states.append(c_state)
        actions.append(action.detach())

        values.append(v_value.detach())

        rewards.append(torch.tensor(rwd))

        log_ps.append(log_p.detach())

        masks.append(torch.tensor(mask))

        c_state = obs

        if done:
            break


    accuracy.append(itr)

    # get prediction for last state, since exited the loop before computing it
    last_v_value = model_critc(torch.FloatTensor(c_state)).detach()


    returns = model_critc.GAE(values,last_v_value ,rewards, masks)

    returns = torch.stack(returns,dim=0)
    log_ps = torch.stack(log_ps,dim=0)
    values = torch.stack(values,dim=0)
    rewards = torch.stack(rewards,dim=0)
    states = torch.stack(states,dim=0)
    actions = torch.stack(actions,dim=0)
    masks = torch.stack(masks,dim=0)
    advantages = returns - values
    advantages = (advantages - torch.mean(advantages))/(torch.std(advantages) + 1e-10)

    # calculate advantages from returns

    actor_c, critic_c = ppo.PPO(states, actions, log_ps, returns, advantages)

    actor_cost+= actor_c
    critic_cost+= critic_c



    if ep % t_printing == 0:

        avr_acc = sum(accuracy)/t_printing
        print("EP: ", ep, " accuracy: ", avr_acc, " critic: ", critic_cost/t_printing, " actor: ", actor_cost/t_printing)

        actor_cost = 0
        critic_cost = 0
        accuracy = []

        training_acc.append(avr_acc)

# Save training accuracy
#torch.save(training_acc,"/Users/michelegaribbo/Desktop/Results/PPO_accuracy.pt")
