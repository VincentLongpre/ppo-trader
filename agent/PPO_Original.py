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

class FeedForwardNN(nn.Module):
    """
    A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, in_dim, out_dim):
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
        self.ln1 = nn.LayerNorm(64)
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

        nn.init.normal_(self.layer1.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer2.weight, mean=0.0, std=0.1)
        nn.init.normal_(self.layer3.weight, mean=0.0, std=0.1)

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

        activation1 = F.tanh(self.ln1(self.layer1(obs)))
        activation2 = F.tanh(self.layer2(activation1))
        output = self.layer3(activation2)

        return output
    
# function that runs each episode
def episode(agent, n_episodes, max_iter = 1000, end_update=True):
    batch_r, batch_s, batch_a = [], [], []

    r_eps = []

    for _ in range(n_episodes):

        s, _ = agent.env.reset()

        termination, truncation = False, False

        a, _ = agent.select_action(torch.tensor(s, dtype=torch.float))

        r_ep = 0

        t = 0

        # while not (termination or truncation):
        for _ in range(max_iter):
            s_prime, r, termination, _, _ = agent.env.step(a)

            a_prime, _ = agent.select_action(torch.tensor(s_prime, dtype=torch.float))

            batch_r.append(r)
            batch_s.append(s)
            batch_a.append(a)

            if not end_update:
                agent.update(s, a , r, s_prime)

            s, a = s_prime, a_prime

            r_ep += r

            t += 1

            if termination:
                break

        r_eps.append(r_ep)

    batch_r, batch_s, batch_a = torch.tensor(batch_r, dtype=torch.float), torch.tensor(batch_s, dtype=torch.float), torch.tensor(batch_a, dtype=torch.float)
    if end_update:
        agent.update(batch_r, batch_s, batch_a)

    return r_eps
    
# function that runs each hyperparameter setting
def hyperparams_run_gradient(agent_class, policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates, n_episodes, max_iter):
    reward_arr_train = np.zeros((len(learning_rates), 50, 1000))

    for i, lr in enumerate(learning_rates):
        for run in range(10): # 50, 1 is for debugging
            print(f'lr_{lr}, for run_{run}')
            agent = agent_class(policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates)

            ep_rewards = []
            for ep in range(100): # 100 is for debugging
                ep_rewards.extend(episode(agent, n_episodes, max_iter, end_update=True))

            reward_arr_train[i, run, :] = ep_rewards

    return reward_arr_train