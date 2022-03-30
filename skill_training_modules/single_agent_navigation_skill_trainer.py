import argparse
import json
import os
from stable_baselines3 import DQN
from .skill_training_environments import SingleAgentNavigationTrainer
from helper_files import (
    MODELS_DIRECTORY,
    LOGS_DIRECTORY,
    STATISTICS_DIRECTORY,
    file_handler,
)
from .training_environment_utils import calculate_optimal_number_of_steps_needed


def train_navigation_skill_with_single_agent(args: argparse.Namespace) -> None:
    max_num_steps = 1.5 * calculate_optimal_number_of_steps_needed(
        tiled_environment_width=args.tiled_environment_width,
        tiled_environment_height=args.tiled_environment_height,
    )

    env = SingleAgentNavigationTrainer(
        max_steps_num=max_num_steps,
        width=args.tiled_environment_width,
        height=args.tiled_environment_height,
    )

    model = DQN(
        "MlpPolicy",
        env,
        verbose=0,
        tensorboard_log=LOGS_DIRECTORY,
    )

    current_model_directory = f"{MODELS_DIRECTORY}/{args.experiment_name}"

    if not os.path.exists(current_model_directory):
        os.makedirs(current_model_directory)

    num_of_cells_visited_per_episode = []

    for i in range(args.num_episodes):
        model.learn(
            total_timesteps=max_num_steps,
            reset_num_timesteps=False,
            tb_log_name=f"{args.experiment_name}",
        )
        model.save(f"{current_model_directory}/{max_num_steps*i}")

        num_of_cells_visited_per_episode.append(len(env.swarm_agent.cells_visited))

    file_handler(
        path=f"{STATISTICS_DIRECTORY}/{args.experiment_name}/num_of_cells_visited_per_episode.txt",
        mode="w",
        func=lambda f: f.write(
            json.dumps(
                {"num_of_cells_visited_per_episode": num_of_cells_visited_per_episode}
            )
        ),
    )
