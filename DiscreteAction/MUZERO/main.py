import os
import torch
import numpy as np
import logging
import gym
from env.hanoi import TowersOfHanoi
from Muzero import Muzero
from utils import setup_logger

def get_env(env_name):
    if env_name == 'Hanoi':
        N = 3  # n. of disks in Tower of Hanoi
        max_steps=200
        env = TowersOfHanoi(N=N,max_steps=max_steps)
        s_space_size = env.oneH_s_size 
        n_action = 6 # n. of action available in each state for Tower of Hanoi (including illegal ones)
    else: # Use for gym env with discrete 1d action space        
        env = gym.make(env_name)
        assert isinstance(env.action_space,gym.spaces.discrete.Discrete), "Must be discrete action space"
        s_space_size = env.observation_space.shape[0]
        n_action = env.action_space.n
        max_steps = env.spec.max_episode_steps
        N = None
    return env, s_space_size, n_action,  max_steps, N


""" Train Muzero for CartPole or Tower of Hanoi environments"""
## ======= Set useful variables ====
s = 1 # Set for performance comparison - change for random seed 
torch.manual_seed(s)
np.random.seed(s)
setup_logger(s)

if torch.cuda.is_available():
    dev=torch.device('cuda')
# Much slower on MAC gpu, probably due to seq. nature of alg. and little paralellisation
#elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available(): ## for MAC GPU usage
#    dev=torch.device('mps')
else:
    dev=torch.device('cpu')

save_results = False

## ======= Select the environment ========
env_n = 1 # 0: 'Hanoi', 1: 'CartPole'
## Set a good set of hyper-params of each env
if env_n ==0:
    env_name = 'Hanoi' 
    discount = 0.8
    n_mcts_simulations = 25 #11 during acting n. of mcts passes for each step
    lr = 0.002
elif env_n == 1:
    env_name = "CartPole-v1" 
    batch_s = 256
    discount = 0.997
    n_mcts_simulations=50
    lr=0.005

## ========= Useful variables: ===========
training_loops = 5000
min_replay_size = 5000
dirichlet_alpha = 0.25
n_ep_x_loop = 1
n_update_x_loop = 1
unroll_n_steps = 5
TD_return = True
n_TD_step = 10
buffer_size = 50000 #int(1e6)
batch_s = 256
priority_replay = True

## ========= Initialise env ========
env, s_space_size, n_action, max_steps, n_disks = get_env(env_name)

## ====== Log command line =====
command_line = f'Env: {env_name}, Training Loops: {training_loops}, Min replay size: {min_replay_size}, lr: {lr}, discount: {discount}, n. MCTS: {n_mcts_simulations}, batch size: {batch_s}, TD_return: {TD_return}, Priority Buff: {priority_replay}'
if env_name == 'Hanoi': # if hanoi also print n. of disks
    command_line += f', N. disks: {n_disks}'
logging.info(command_line)

## ======== Initialise alg. ========
muzero = Muzero(env=env, s_space_size=s_space_size, n_action=n_action, discount=discount, dirichlet_alpha=dirichlet_alpha, n_mcts_simulations=n_mcts_simulations, unroll_n_steps=unroll_n_steps, batch_s=batch_s, TD_return=TD_return,n_TD_step=n_TD_step, lr=lr, buffer_size=buffer_size, priority_replay=priority_replay, device=dev, n_ep_x_loop=n_ep_x_loop, n_update_x_loop=n_update_x_loop)

## ======== Run training ==========
tot_acc = muzero.training_loop(training_loops, min_replay_size)

## ===== Save results =========
file_indx = 1 # label to denote different runs

# Create directories to store results
file_dir = os.path.dirname(os.path.abspath(__file__)) # get path of current file
file_dir = os.path.join(file_dir,'results',str(file_indx))
acc_dir = os.path.join(file_dir,'training_accuracy.pt')
model_dir = os.path.join(file_dir,'muzero_model.pt')

if save_results:
    # Create directory if it did't exist before
    os.makedirs(file_dir, exist_ok=True)

    # Store command line
    with open(os.path.join(file_dir,'commands.txt'), 'w') as f:
        f.write(command_line)
    # Store accuracy
    torch.save(torch.tensor(tot_acc),acc_dir)
    # Store model
    torch.save({
        'Muzero_net': muzero.networks.state_dict(),
        'Net_optim': muzero.networks.optimiser.state_dict()
    }, model_dir)
