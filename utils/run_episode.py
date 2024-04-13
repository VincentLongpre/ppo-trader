import numpy as np
import torch
import json
import os

# function that runs each episode
def episode(agent, n_batch, max_iter = 1000, testing=False):
    r_eps = []

    for _ in range(n_batch):

        batch_r, batch_s, batch_a, batch_terminal = [], [], [], []

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
            batch_terminal.append(termination)

            s, a = s_prime, a_prime

            r_ep += r

            t += 1

            if termination:
                break

        r_eps.append(r_ep)
        # print(f'actions are: {batch_a[-1]}')
        # print(f'Variance is: {agent.cov_var}')
        batch_r, batch_s, batch_a, batch_terminal = torch.tensor(np.array(batch_r), dtype=torch.float), torch.tensor(np.array(batch_s), dtype=torch.float), torch.tensor(np.array(batch_a), dtype=torch.float), torch.tensor(np.array(batch_terminal), dtype=torch.float)

        if not testing:
            agent.update(batch_r, batch_s, batch_a, batch_terminal)

    return r_eps

# function that runs each hyperparameter setting
def hyperparams_run_gradient(agent_class, policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates, n_episodes, max_iter):
    reward_arr_train = np.zeros((len(learning_rates), 50, 1000))

    for i, lr in enumerate(learning_rates):
        for run in range(10): # 50, 1 is for debugging
            print(f'lr_{lr}, for run_{run}')
            agent = agent_class(policy_class, env, lr, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates)

            ep_rewards = []
            for ep in range(1000): # 100 is for debugging
                print(ep)
                ep_rewards.extend(episode(agent, n_episodes, max_iter))

            reward_arr_train[i, run, :] = ep_rewards

    return reward_arr_train

def run_trials(agent_class, policy_class, env, save_path, model_name, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates, n_episodes, max_iter):
    os.makedirs(save_path, exist_ok=True)

    for run in range(10): # 50, 1 is for debugging
        reward_arr_train = []

        for _ in range(3):
            try:
                agent = agent_class(policy_class, env, learning_rates, gamma, clip, ent_coef, critic_factor, max_grad_norm, gae_lambda, n_updates)

                for ep in range(120): # 100 is for debugging
                    ep_returns = episode(agent, n_episodes, max_iter)
                    reward_arr_train.extend(ep_returns)

                    if ep % 100 == 0:
                        print(f"Episode {ep} - Mean Return: {np.mean(ep_returns)}")

                reward_arr_train = np.array(reward_arr_train)
                with open(save_path + f"{model_name}_{run}.json", 'w') as f:
                    json.dump(reward_arr_train.tolist(), f)

                break
            
            except:
                pass

    return reward_arr_train
