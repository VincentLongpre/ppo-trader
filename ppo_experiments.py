import gymnasium as gym
import yaml
import json
import numpy as np
from agent.PPO import MCPPO, FeedForwardNN, run_trials
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

class RewardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardCallback, self).__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Retrieve the current reward and append it to the rewards list
        current_reward = self.model.ep_info_buffer[0]["r"]
        self.rewards.append(current_reward)
        return True

if __name__ == "__main__":
    save_path = "runs/pendulum/"
    env_name = 'Pendulum-v1'
    env = gym.make(env_name)

    with open("configs/ppo_configs.yaml", 'r') as f:
        ppo_configs = yaml.safe_load(f)

    with open("configs/env_configs.yaml", 'r') as f:
        env_configs = yaml.safe_load(f)

    run_trials(MCPPO, FeedForwardNN, env, save_path, **ppo_configs)

    for run in range(10):
        model = PPO('MlpPolicy', env, verbose=1)

        reward_callback = RewardCallback()

        model.learn(total_timesteps=200000, callback=reward_callback, progress_bar=True)

        reward_arr_train = np.array(reward_callback.rewards).tolist()

        with open(save_path + f"ppo_{run}.json", 'w') as f:
            json.dump(reward_arr_train, f)

    