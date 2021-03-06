import sys
import os
from typing import Dict
import wandb

root_dir = os.path.dirname(os.path.realpath(__file__))

for i in range(3):
    root_dir = os.path.dirname(root_dir)
    sys.path.append(root_dir)

from skill_training_modules import (
    SingleAgentNavigationTrainer,
    MultiAgentNavigationTrainer,
    SenseBroadcastTrainer,
    CommitToOpinionTrainer,
    DynamicOpinionWeightingTrainer,
)

from helper_files import TRAINED_MODELS_DIRECTORY

EPISODES = 3
ITERATIONS_PER_EPISODE = 5

env_name_to_class_map = {
    "single_agent_navigation": SingleAgentNavigationTrainer,
    "multi_agent_navigation": MultiAgentNavigationTrainer,
    "sense_broadcast": SenseBroadcastTrainer,
    "commit_to_opinion": CommitToOpinionTrainer,
    "dynamic_opinion_weighting": DynamicOpinionWeightingTrainer,
}


def generic_env_check(env_name: str, env_configs: Dict[str, int]):
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env

    wandb.init(mode="disabled")

    env = env_name_to_class_map[env_name](**env_configs)

    if env_configs["model"]:
        model = PPO.load(f"{TRAINED_MODELS_DIRECTORY}/{env_configs['model']}")
        env.set_model(model)

    check_env(env)

    for _ in range(EPISODES):
        print("--- Reset ---")
        print("First observations", env.reset())
        done = False
        while not (done):
            random_action = env.action_space.sample()
            print("action", random_action)
            obs, reward, done, _ = env.step(random_action)
            print("observation", obs)
            print("reward", reward)
