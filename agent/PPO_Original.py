import gymnasium as gym
import numpy as np
import torch
import yaml
import json
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import MultivariateNormal

class PPO:

    def __init__(self, policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates):

        self.lr = lr                                # Learning rate of actor optimizer
        self.gamma = gamma                          # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = clip
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.actor = policy_class(self.s_dim, self.a_dim)
        self.critic = policy_class(self.s_dim, 1)

        self.actor_optim = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=self.lr)

        self.cov_var = torch.full(size=(self.a_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def select_action(self, s):

        mean = self.actor(s)

        dist = MultivariateNormal(mean, self.cov_mat)

        a = dist.sample()

        log_prob = dist.log_prob(a)

        return a.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_s, batch_a):

        V = self.critic(batch_s).squeeze()

        mean = self.actor(batch_s)
        dist = MultivariateNormal(mean, self.cov_mat)
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

            actor_loss = (-torch.min(term1, term2)).mean()
            critic_loss = nn.MSELoss()(V, batch_G)

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()