import argparse
import os
from stable_baselines3 import DQN
from .skill_training_environments import SingleAgentNavigationTrainer
from helper_files import MODELS_DIRECTORY, LOGS_DIRECTORY


def train_navigation_skill_with_single_agent(args: argparse.Namespace) -> None:
    env = SingleAgentNavigationTrainer(max_steps_num=args.num_steps)
    env.reset()

    model = DQN(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=LOGS_DIRECTORY,
    )

    current_model_directory = f"{MODELS_DIRECTORY}/{args.experiment_name}"

    if not os.path.exists(current_model_directory):
        os.makedirs(current_model_directory)

    num_of_cell_visited_per_episode = []

    for i in range(args.num_episodes):
        print("beginning episode: ", i)
        model.learn(
            total_timesteps=args.num_steps,
            reset_num_timesteps=False,
            tb_log_name=f"{args.experiment_name}",
        )
        model.save(f"{current_model_directory}/{args.num_steps*i}")

        num_of_cell_visited_per_episode.append(len(env.swarm_agent.cells_visited))
        print(num_of_cell_visited_per_episode)
        print("end episode: ", i)
