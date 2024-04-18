import numpy as np
import pickle as pkl
import torch
import json
import os

def episode(agent, n_batch, max_iter = 10000, testing=False):
    """
    Run a batch of episodes for the given agent.

    Parameters:
    - agent (object): The agent instance being trained.
    - n_batch (int): Number of episodes to run in the batch.
    - max_iter (int): The maximum number of iterations (steps) allowed for each episode. Default is 10000.
    - testing (bool): Whether the episodes are for testing purposes. Default is False.

    Returns:
    - r_eps (list): List of episodic returns for each episode in the batch.
    """
    r_eps = []
    if testing:
        asset_hist = [agent.env.asset_memory[0]]

    for _ in range(n_batch):

        batch_r, batch_s, batch_a, batch_terminal = [], [], [], []

        s, _ = agent.env.reset()

        termination, truncation = False, False

        a, _ = agent.select_action(torch.tensor(s, dtype=torch.float))

        r_ep = 0

        t = 0

        # while not (termination or truncation):
        for i in range(max_iter):
            s_prime, r, termination, _, _ = agent.env.step(a)

            a_prime, _ = agent.select_action(torch.tensor(s_prime, dtype=torch.float))

            batch_r.append(r)
            batch_s.append(s)
            batch_a.append(a)
            batch_terminal.append(termination)

            s, a = s_prime, a_prime

            r_ep += r

            t += 1

            if termination:
                break

            if testing:
                asset_hist.append(agent.env.asset_memory[-1])

        if not testing:
            r_eps.append(r_ep)

        batch_r, batch_s, batch_a, batch_terminal = torch.tensor(np.array(batch_r), dtype=torch.float), torch.tensor(np.array(batch_s), dtype=torch.float), torch.tensor(np.array(batch_a), dtype=torch.float), torch.tensor(np.array(batch_terminal), dtype=torch.float)

        if not testing:
            agent.update(batch_r, batch_s, batch_a, batch_terminal)

    if testing:
        return asset_hist

    return r_eps

def run_trials(agent_class, policy_class, env, run_save_path, model_save_path, model_name, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates, n_episodes, max_iter):
    """
    Run multiple trials for training the agent on the given environment and save the resulting returns 
    (and models if a model save path is specified).

    Parameters:
    - agent_class (object): Class of the agent to be trained.
    - policy_class (object): Class of the policy used by the agent.
    - env (object): Environment for training the agent.
    - run_save_path (str): Path to save the training results.
    - model_save_path (str): Path to save the trained models. Default is None.
    - model_name (str): Name of the model being trained.
    - learning_rate (float): Learning rate for training the agent.
    - gamma (float): Discount factor for future rewards.
    - clip (float): Clip parameter for PPO.
    - ent_coef (float): Coefficient for the entropy loss.
    - critic_factor (float): Factor for critic loss in PPO.
    - max_grad_norm (float): Maximum gradient norm for PPO.
    - gae_lambda (float): Lambda value for generalized advantage estimation (GAE).
    - n_updates (int): Number of updates per episode for PPO.
    - n_episodes (int): Number of episodes per trial.
    - max_iter (int): Maximum number of iterations (steps) per episode.

    Returns:
    - reward_arr_train (array): Array containing the episodic returns for each episode in the last trial.
    """
    os.makedirs(run_save_path, exist_ok=True)
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)

    for run in range(10):
        for _ in range(3):
            reward_arr_train = []
            try:
                agent = agent_class(policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates)

                for ep in range(1000):
                    ep_returns = episode(agent, n_episodes, max_iter)
                    reward_arr_train.extend(ep_returns)

                    if ep % 10 == 0:
                        print(f"Episode {ep} - Mean Return: {np.mean(ep_returns)}")

                reward_arr_train = np.array(reward_arr_train)

                with open(run_save_path + f"{model_name}_{run}.json", 'w') as f:
                    json.dump(reward_arr_train.tolist(), f)

                if model_save_path: 
                    with open(model_save_path + f"{model_name}/{run}.pkl", 'wb') as f:
                        pkl.dump(agent, f)
                break
            
            except:
                continue

    return reward_arr_train
