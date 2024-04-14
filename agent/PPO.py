import gymnasium as gym
import numpy as np
import torch
import yaml
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from utils.run_episode import hyperparams_run_gradient
import logging

logging.basicConfig(filename='training.log', level=logging.INFO, format='%(message)s')

class PPO:
    def __init__(self, policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lamda, n_updates):
        self.lr = lr                                # Learning rate of actor optimizer
        self.gamma = gamma                          # Discount factor to be applied when calculating Rewards-To-Go
        self.clip = clip
        self.ent_coef = ent_coef
        self.critic_factor = critic_factor
        self.max_grad_norm = max_grad_norm
        self.gae_lambda = gae_lamda
        self.n_updates = n_updates

        self.env = env
        self.s_dim = env.observation_space.shape[0]
        self.a_dim = env.action_space.shape[0]

        self.actor = policy_class(self.s_dim, self.a_dim)
        self.critic = policy_class(self.s_dim, 1)

        self.cov_var = nn.Parameter(torch.full(size=(self.a_dim,), fill_value=1.0))
        self.cov_mat = torch.diag(self.cov_var)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()) + [self.cov_var],
            lr=lr
        )

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
        entropy = dist.entropy()

        return V, log_prob, entropy

    def compute_G(self, batch_r, batch_terminal, V):
        V = V.clone().cpu().detach().numpy().flatten()

        last_gae_lam = 0
        batch_size = len(batch_r)

        A = [0]*batch_size

        for step in reversed(range(batch_size)):

            if step == batch_size - 1:
                next_non_terminal = 0
                next_value = 0
            else:
                next_non_terminal = 1.0 - batch_terminal[step + 1]
                next_value = V[step + 1]

            delta = batch_r[step] + self.gamma * next_value * next_non_terminal - V[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            A[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        G = A + V

        G = torch.tensor(G, dtype=torch.float)
        A = torch.tensor(A, dtype=torch.float)

        return G, A

    def update(self, batch_r, batch_s, batch_a, batch_terminal):
        V, old_log_prob, entropy = self.evaluate(batch_s, batch_a)

        old_log_prob = old_log_prob.detach()

        batch_G, A = self.compute_G(batch_r, batch_terminal, V)

        logging.info("Mean Advantage: %.4f", A.mean().item())
        logging.info("Std Advantage: %.4f", A.std().item())

        A = (A - A.mean()) / (A.std() + 1e-10)

        for _ in range(self.n_updates):
            V, log_prob, entropy = self.evaluate(batch_s, batch_a)

            ratios = torch.exp(log_prob - old_log_prob)

            term1 = ratios * A
            term2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A

            actor_loss = (-torch.min(term1, term2)).mean()
            critic_loss = nn.MSELoss()(V, batch_G)
            loss = actor_loss + self.critic_factor * critic_loss + self.ent_coef * entropy.mean()

            logging.info("Log Probabilities: %s", log_prob.tolist())
            logging.info("Mean Ratios: %.4f", torch.mean(ratios).item())
            logging.info("Actor Loss: %.4f", actor_loss.item())
            logging.info("Critic Loss: %.4f", critic_loss.item())
            logging.info("Total Loss: %.4f", loss.item())
            logging.info("Var: %.4f", torch.mean(self.cov_var).item())

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(list(self.actor.parameters()) + list(self.critic.parameters()) + [self.cov_var], self.max_grad_norm)
            self.optimizer.step()
"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

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

        # Initialize weights with low values
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

        activation1 = F.tanh(self.ln1(self.layer1(obs))) # self.ln1(
        activation2 = F.tanh(self.layer2(activation1))
        output = self.layer3(activation2) # F.tanh()

        return output

if __name__ == "__main__":
    from stable_baselines3 import PPO as BPPO
    env_name = 'Pendulum-v1' # 'CartPole-v1' # 'MountainCar-v0'
    env = gym.make(env_name)

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    ppo_configs['learning_rates'] = [0.0003]

    hyperparams_run_gradient(PPO, FeedForwardNN, env, **ppo_configs)
    # model = BPPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=200000, progress_bar=True)