import numpy as np
import torch
import json

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

def run_trials(agent_class, policy_class, env, save_path, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates, n_episodes, max_iter):
    
    for run in range(10): # 50, 1 is for debugging
        reward_arr_train = []
        agent = agent_class(policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, n_updates)

        for ep in range(500): # 100 is for debugging
            reward_arr_train.extend(episode(agent, n_episodes, max_iter, end_update=True))

        reward_arr_train = np.array(reward_arr_train)
        with open(save_path + f"mcppo_{run}.json", 'w') as f:
            json.dump(reward_arr_train.tolist(), f)

    return reward_arr_train
