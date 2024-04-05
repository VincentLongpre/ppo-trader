import gymnasium as gym
import time

import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:

    def __init__(self, policy_class, env, lr, gamma, clip, n_updates):

        self.lr = lr                                # Learning rate of actor optimizer
        self.gamma = gamma                          # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = clip
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.actor_critic = policy_class(self.s_dim, self.a_dim, 1)

        self.actor_critic_optim = Adam(self.actor_critic.parameters(), lr=self.lr)
        # self.critic_optim = Adam(self.actor_critic.parameters(), lr=self.lr)

        self.cov_var_actor = torch.ones(size=(self.a_dim,)) # self.actor_critic.log_std_actor.exp()
        self.cov_mat_actor = torch.diag(self.cov_var_actor)

        # self.cov_var_critic = torch.full(size=(self.a_dim,), fill_value=self.actor_critic.log_std_critic)
        # self.cov_mat_critic = torch.diag(self.cov_var_critic)

    def select_action(self, s):

        mean,_ = self.actor_critic(s)

        self.cov_var_actor = torch.ones(size=(self.a_dim,)) # * self.actor_critic.log_std_actor.exp()
        self.cov_mat_actor = torch.diag(self.cov_var_actor)

        dist = MultivariateNormal(mean, self.cov_mat_actor)

        a = torch.clamp(dist.sample(), min=-1, max=1) # torch.clamp(, min=-1, max=1)

        log_prob = dist.log_prob(a)

        return a.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_s, batch_a):

        V = self.actor_critic(batch_s)[1].squeeze()

        mean,_ = self.actor_critic(batch_s)

        self.cov_var_actor = torch.ones(size=(self.a_dim,)) # * self.actor_critic.log_std_actor.exp()
        self.cov_mat_actor = torch.diag(self.cov_var_actor)

        dist = MultivariateNormal(mean, self.cov_mat_actor)
        log_prob = dist.log_prob(batch_a)

        return V, log_prob

    def compute_G(self, batch_r):

        G = 0
        batch_G = []

        for r in reversed(batch_r):

            G = r + self.gamma*G
            batch_G.insert(0, G)

        batch_G = torch.tensor(batch_G, dtype=torch.float)

        return batch_G

    def update(self, batch_r, batch_s, batch_a):

        V, old_log_prob = self.evaluate(batch_s, batch_a)

        old_log_prob = old_log_prob.detach()

        batch_G = self.compute_G(batch_r)

        A = batch_G - V.detach()

        A = (A - A.mean())/(A.std() + 1e-10)

        for i in range(self.n_updates):

            V, log_prob = self.evaluate(batch_s, batch_a)

            ratios = torch.exp(log_prob - old_log_prob)

            term1 = ratios*A
            term2 = torch.clamp(ratios, 1-self.clip, 1+self.clip)*A

            actor_critic_loss = (-torch.min(term1, term2)).mean() + nn.MSELoss()(V, batch_G)

            self.actor_critic_optim.zero_grad()
            actor_critic_loss.backward(retain_graph=True)
            self.actor_critic_optim.step()

            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()

"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class FeedForwardNN(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim_actor, out_dim_critic):
        """
        Initialize the network and set up the layers.

        Parameters:
            in_dim - input dimensions as an int
            out_dim - output dimensions as an int

        Return:
            None
        """
        super(FeedForwardNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3_actor = nn.Linear(64, out_dim_actor)
        self.layer3_critic = nn.Linear(64, out_dim_critic)

        # self.log_std_actor = nn.Parameter(torch.zeros(out_dim_actor), requires_grad=True)
        # self.log_std_critic = nn.Parameter(torch.ones(self.action_dim), requires_grad=True)

    def forward(self, obs):
        """
            Runs a forward pass on the neural network.

            Parameters:
                obs - observation to pass as input

            Return:
                output - the output of our forward pass
        """
        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)

        activation1 = F.relu(self.layer1(obs))
        activation2 = F.relu(self.layer2(activation1))
        output_actor = self.layer3_actor(activation2) # F.tanh()
        output_critic = self.layer3_critic(activation2)

        return output_actor, output_critic

# function that runs each episode
def episode(agent, n_batch, max_iter = 1000):

    batch_r, batch_s, batch_a = [], [], []

    r_eps = []

    for i in range(n_batch):

        (s, info) = agent.env.reset()

        termination, truncation = False, False

        a, _ = agent.select_action(torch.tensor(s, dtype=torch.float))

        r_ep = 0

        t = 0

        # while not (termination or truncation):
        for j in range(max_iter):

            s_prime, r, termination, truncation, info = agent.env.step(a)

            a_prime, _ = agent.select_action(torch.tensor(s_prime, dtype=torch.float))

            batch_r.append(r)
            batch_s.append(s)
            batch_a.append(a)

            s, a = s_prime, a_prime

            r_ep += r

            t += 1

            if termination:
                break

        r_eps.append(r_ep)

    batch_r, batch_s, batch_a = torch.tensor(np.array(batch_r), dtype=torch.float), torch.tensor(np.array(batch_s), dtype=torch.float), torch.tensor(np.array(batch_a), dtype=torch.float)
    agent.update(batch_r, batch_s, batch_a)

    return np.mean(r_eps)

# function that runs each hyperparameter setting
def hyperparams_run_gradient(policy_class, env, learning_rates, gamma, clip, n_updates, n_batch, max_iter=1000):

    reward_arr_train = np.zeros((len(learning_rates), 50, 1000))

    for i, lr in enumerate(learning_rates):

        for run in range(1): # 50, 1 is for debugging
            print(f'temp_{lr}, for run_{run}')

            agent = PPO(policy_class, env, lr, gamma, clip, n_updates)

            for ep in range(100): # 100 is for debugging

                reward_arr_train[i, run, ep] = episode(agent, n_batch, max_iter)
                print(reward_arr_train[i, run, ep])

    return reward_arr_train

if __name__ == "__main__":
    env_name = 'Pendulum-v1' # 'CartPole-v1' # 'MountainCar-v0'
    env = gym.make(env_name)

    learning_rates = [3e-4]
    gamma = 0.99
    clip = 0.2
    n_updates = 10
    n_batch = 10
    max_iter = 200

    policy_class = FeedForwardNN

    torch.autograd.set_detect_anomaly(True)

    reward_arr_train = hyperparams_run_gradient(policy_class, env, learning_rates, gamma, clip, n_updates, n_batch, max_iter=max_iter)