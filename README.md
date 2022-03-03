# LunarLander-using-REINFORCE
- REINFORCE belongs to a special class of Reinforcement Learning algorithms called Policy Gradient algorithms. A simple implementation of this algorithm would involve creating a Policy: a model that takes a state as input and generates the probability of taking an action as output. A policy is essentially a guide or cheat-sheet for the agent telling it what action to take at each state. The policy is then iterated on and tweaked slightly at each step until we get a policy that solves the environment.
-  The policy is usually a Neural Network that takes the state as input and generates a probability distribution across action space as output.

## Reinforce Algorithm
- Each policy generates the probability of taking an action in each state of the environment.
- The agent samples from these probabilities and selects an action to perform in the environment. At the end of an episode, we know the total rewards the agent can get if it follows that policy. We backpropagate the reward through the path the agent took to estimate the “Expected reward” at each state for a given policy.

  <img src="https://github.com/BhanuPrakashPebbeti/LunarLander-using-REINFORCE/blob/main/assets/Policy_trajectories.png" width="800" height="400">

- Discounted reward is the sum of all the rewards the agent receives in that future discounted by a factor Gamma.
- The discounted reward at any stage is the reward it receives at the next step + a discounted sum of all rewards the agent receives in the future.

  <img src="https://github.com/BhanuPrakashPebbeti/LunarLander-using-REINFORCE/blob/main/assets/Policy_trajectories_2.png" width="800" height="400">


## Algorithm steps
The steps involved in the implementation of REINFORCE would be as follows:
- Initialize a Random Policy (a NN that takes the state as input and returns the probability of actions)
- Use the policy to play N steps of the game — record action probabilities-from policy, reward-from environment, action — sampled by agent
- Calculate the discounted reward for each step by backpropagation
- Calculate expected reward G
- Adjust weights of Policy (back-propagate error in NN) to increase G
- Repeat from 2

  <img src="https://github.com/BhanuPrakashPebbeti/LunarLander-using-REINFORCE/blob/main/assets/Reinforce%20Algorithm.png" width="800" height="300">

## Reward Stats while Training

<img src="https://github.com/BhanuPrakashPebbeti/LunarLander-using-REINFORCE/blob/main/Rewards.png" width="400" height="400">

## Lunar Lander
![LunarLander_gif](https://github.com/BhanuPrakashPebbeti/LunarLander-using-REINFORCE/blob/main/assets/LunarLander.gif)
