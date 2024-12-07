# 3D Bin Packing using Deep Reinforcement Learning (DRL).

This strategy aims to solve the problem using a <b> deep reinforcement learning </b> approach.

<b> Reinforcement learning (RL) </b> is a machine learning approach where an agent learns to make optimal decisions in an environment by maximizing rewards. Recently, RL has been applied to the bin packing problem, offering a promising alternative to traditional methods despite computational limitations. The experimental results demonstrate RL's potential to tackle complex resource allocation challenges more efficiently.

We utilize two models for the DRL strategy:

1. <b> Proximal Policy Optimization (PPO) </b>
2. <b> Advantage Actor Critic (A2C) </b>

Although the strategy can be configured to use the A2C model by setting the `model` parameter in the `drl.config` file as `A2C`, we will focus on the PPO model in this implementation. Due the structure of the PPO model, it is more suited for the bin packing problem than the A2C model.

In this implementation, the <b> stable-baselines3 </b> library is used to implement the PPO and A2C models.

We use the concept of height maps to represent the state of the bin. This is a more efficient way to represent the state of the bin compared to the one-hot encoding used in the traditional DRL strategy.

We encode the state of each package into a vector of size 7, representing the width, depth and height, weight, priority, delay cost and if it has been packed or not.

The models uses a <b> MlpPolicy </b> for the policy network. And uses masking of the action to prevent the agent from selecting invalid actions.

This invalid actions result in a lower score for the agent.

While the taking a correct action results in a higher score for the agent, the model is also encouraged to take actions that would result in a lower number of items remaining in the bin. This is done by giving a negative reward for each item that remains in the bin.

The score is structured in a way that encourages the model to take actions that would result in a lower height in the bin. This is done by giving a negative reward for each level that the item is placed higher in the bin. As well as encourage the model to place items that results in a compact packing in the bin. The score is also influnced by other factors allowing the model to learn a more global policy for the bin packing problem.

## How to Run

To run the drl strategy and solve the 3D Bin Packing Problem, follow the steps below:

### Method 1: Modify the Configuration

Ensure that the main.config file is correctly set up by changing the default_strategy to drl. This will configure the optimizer to use the DRL strategy.

### Method 2: Using the command line

bash
python src/main.py -s drl -d

## drl_strategy.py

This file contains the main class for the DRL strategy. This class is used to run the DRL strategy.

## drl_utils.py

This file contains all functions and classes definition along with the model definition for the DRL strategy.

## drl.config

This file contains the configuration for the DRL strategy. Can be used to set model parameters like the total number of timesteps, the learning rate, etc.

Parameters defined in the `drl.config` file: 1. `model` (`str`): The model to use for the DRL strategy. Can be either `PPO` or `A2C`. <br> 2. `total_timesteps` (`int`): The total number of timesteps to train the model. <br> 3. `learning_rate` (`float`): The learning rate for the model. <br> 4. `ent_coef` (`float`): The entropy coefficient for the model. <br> 5. `use_sde` (`bool`): Whether to use the Stochastic Gradient Descent estimator. <br> 6. `sde_sample_freq` (`int`): The frequency at which to update the covariance matrix of the Ornstein-Uhlenbeck process in th score function. <br> 7. `gamma` (`float`): The discount factor for the model.
