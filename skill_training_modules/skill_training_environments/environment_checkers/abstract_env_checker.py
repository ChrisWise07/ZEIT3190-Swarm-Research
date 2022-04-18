import sys
import os
import wandb
from typing import Dict

root_dir = os.path.dirname(os.path.realpath(__file__))

for i in range(3):
    root_dir = os.path.dirname(root_dir)
    sys.path.append(root_dir)

from skill_training_modules import (
    SingleAgentNavigationTrainer,
    MultiAgentNavigationTrainer,
    SenseBroadcastTrainer,
)
from stable_baselines3.common.env_checker import check_env

EPISODES = 3
ITERATIONS_PER_EPISODE = 5

env_name_to_class_map = {
    "single_agent_navigation": SingleAgentNavigationTrainer,
    "multi_agent_navigation": MultiAgentNavigationTrainer,
    "sense_broadcast": SenseBroadcastTrainer,
}


def generic_env_check(env_name: str, env_configs: Dict[str, int]):
    wandb.init(mode="disabled")

    env = env_name_to_class_map[env_name](**env_configs)
    check_env(env)

    for _ in range(EPISODES):
        print("--- Reset ---")
        env.reset()
        done = False
        while not (done):
            random_action = env.action_space.sample()
            print("action", random_action)
            obs, reward, done, _ = env.step(random_action)
            print("observation", obs)
            print("reward", reward)
