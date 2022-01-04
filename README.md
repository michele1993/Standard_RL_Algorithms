# Implementation of standard deep RL algorithms

This repository contains the implementation of some of the standard model-free deep reinforcement learning algorithms, divided in discrete and continous action spaces. These algorithms include Deep-Q-Learning, REINFORCE (with a baseline), a TD0 actor-critic algorithms (AC2) and PPO for the OpenAI Cartpole (discte action space). There are also a DDPG and a Twin Delayed DDPG (TD3) implementation for the OpenAI Inverted Pendulum and BipedalWalker environments respectively (continous action space).   

For the sake of practice, I compared the performance of the algorithms on the Cartpole environment for the discrete action case. However, this is no accurate comparison since I did not run any thorough hyper-parameter search for each algorithms, nor have I compared performance across multiple random seeds (i.e. all performance based on a single random seed). 


![This is an image](/DiscreteAction/Images/CartPole_Comparison.png)
![This is an image](/ContinousAction/Images/DDPG_acc.png)

