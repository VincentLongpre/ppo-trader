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
        self.ent_coef = ent_coef
        self.critic_factor = critic_factor
        self.max_grad_norm = max_grad_norm
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

        #change to low and high bound
        low_bound = self.env.action_space.low[0]
        high_bound = self.env.action_space.high[0]
        a = torch.clamp(a, low_bound, high_bound)

        return a.detach().numpy(), log_prob.detach()

    def evaluate(self, batch_s, batch_a):
        V = self.critic(batch_s).squeeze()

        mean = self.actor(batch_s)
        dist = MultivariateNormal(mean, self.cov_mat)

        log_prob = dist.log_prob(batch_a)
        entropy = dist.entropy()

        return V, log_prob, entropy

class MCPPO(PPO):
    def compute_G(self, batch_r):
        G = 0
        batch_G = []

        for r in reversed(batch_r):
            G = r + self.gamma * G
            batch_G.insert(0, G)

        batch_G = torch.tensor(batch_G, dtype=torch.float)

        return batch_G

    def update(self, batch_r, batch_s, batch_a):
        V, old_log_prob, entropy = self.evaluate(batch_s, batch_a)

        old_log_prob = old_log_prob.detach()

        batch_G = self.compute_G(batch_r)

        A = batch_G - V.detach()

        A = (A - A.mean())/(A.std() + 1e-10)

        for _ in range(self.n_updates):
            V, log_prob, entropy = self.evaluate(batch_s, batch_a)

            ratios = torch.exp(log_prob - old_log_prob)

            term1 = ratios * A
            term2 = torch.clamp(ratios, 1-self.clip, 1+self.clip) * A

            actor_loss = (-torch.min(term1, term2)).mean()
            critic_loss = nn.MSELoss()(V, batch_G)
            loss = actor_loss + self.critic_factor * critic_loss + self.ent_coef * entropy.mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            self.optimizer.step()

            # self.actor_optim.zero_grad()
            # actor_loss.backward()
            # self.actor_optim.step()

            # self.critic_optim.zero_grad()
            # critic_loss.backward()
            # self.critic_optim.step()

class TDPPO(PPO):
    def update(self, s, a, r, next_s):
        r = torch.tensor(r, dtype=torch.float)
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)
        next_s = torch.tensor(next_s, dtype=torch.float)

        V, old_log_prob = self.evaluate(s, a)
        old_log_prob = old_log_prob.detach()

        with torch.no_grad():
            next_V = self.critic(next_s).squeeze()
            TD_target = r + self.gamma * next_V

        critic_loss = nn.MSELoss()(V, TD_target.detach())

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        A = TD_target - V.detach()

        for _ in range(self.n_updates):
            V, log_prob = self.evaluate(s, a)
            ratios = torch.exp(log_prob - old_log_prob)
            term1 = ratios * A
            term2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A
            actor_loss = (-torch.min(term1, term2)).mean()

            self.actor_optim.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_optim.step()

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
        self.layer2 = nn.Linear(64, 64)
        self.layer3 = nn.Linear(64, out_dim)

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

    return np.mean(r_eps)

# function that runs each hyperparameter setting
def hyperparams_run_gradient(agent_class, policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates, n_episodes, max_iter):
    reward_arr_train = np.zeros((len(learning_rates), 50, 1000))

    for i, lr in enumerate(learning_rates):
        for run in range(10): # 50, 1 is for debugging
            print(f'lr_{lr}, for run_{run}')
            agent = agent_class(policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates)

            for ep in range(100): # 100 is for debugging
                reward_arr_train[i, run, ep] = episode(agent, n_episodes, max_iter, end_update=True)
                print(reward_arr_train[i, run, ep])

    return reward_arr_train

def run_trials(agent_class, policy_class, env, save_path, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates, n_episodes, max_iter):
    
    for run in range(10): # 50, 1 is for debugging
        reward_arr_train = np.zeros((100))
        agent = agent_class(policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates)

        for ep in range(100): # 100 is for debugging
            reward_arr_train[ep] = episode(agent, n_episodes, max_iter, end_update=True)

        with open(save_path + f"mcppo_{run}.json", 'w') as f:
            json.dump(reward_arr_train.tolist(), f)

    return reward_arr_train

if __name__ == "__main__":
    from stable_baselines3 import PPO
    env_name = 'Pendulum-v1' # 'CartPole-v1' # 'MountainCar-v0'
    env = gym.make(env_name)

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    ppo_configs['learning_rates'] = [0.0003]

    hyperparams_run_gradient(MCPPO, FeedForwardNN, env, **ppo_configs)
    # model = PPO('MlpPolicy', env, verbose=1)
    # model.learn(total_timesteps=200000, progress_bar=True)