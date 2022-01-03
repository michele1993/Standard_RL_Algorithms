# Implementation of standard deep reinforcement learning algorithms

This repository contains the implementation of some of the standard deep reinforcement learning algorithms, divided in discrete and continous action spaces. These algorithms included Deep-Q-Learning, REINFORCE (with a baseline), a TD0 actor-critic algorithms (AC2) and PPO for the OpenAI Cartpole. There is also a DDPG implementation for the OpenAI Inverted Pendulum environment.  

For the sake of practice, I compared the performance of the algorithms on the Cartpole environment. However, this is no accurate comparison since I did not run any thorough hyper-parameter search for each algorithms, nor have I compared performance across multiple random seeds (i.e. all performance based on a single random seed). 


![This is an image](/DiscreteAction/Images/CartPole_Comparison.png)
