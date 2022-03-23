import argparse
import json
import time
import os
from typing import Callable
from constants import EXPERIMENT_DATA_ROOT_DIRECTORY
from testing_and_training_modules import file_handler
from experiment_modules_map import experiment_modules_map

parser = argparse.ArgumentParser(description="Experiment hyperparameters & settings")

parser.add_argument(
    "--is_clustered",
    type=bool,
    default=False,
    help="Controls if the tiled environment is clustered (default=False)",
)
parser.add_argument(
    "--initial_observations_useful",
    type=bool,
    default=True,
    help=(
        "Controls if a cluster tiled environment has white tile first (default=True) "
        "Note this is ignored in a non-clustered environment."
    ),
)
parser.add_argument(
    "--tiled_environment_height",
    type=int,
    default=20,
    help="The height of the tiled environment i.e. the number of rows (default=20)",
)
parser.add_argument(
    "--tiled_environment_width",
    type=int,
    default=20,
    help="The width of the tiled environment i.e. the number of columns (default=20)",
)
parser.add_argument(
    "--num_of_swarm_agents",
    type=int,
    default=20,
    help="The number of swarm agents (default=20)",
)
parser.add_argument(
    "--ratio_of_white_to_black_tiles",
    type=float,
    default=0.5,
    help=(
        "The ratio of white to black tiles (default=0.5) "
        "Note that the tiled environment may not have this exact ratio. "
        "Therefore the actual ratio will be calculated first "
        "and swarm ratio accuracy is measured against this measured ratio."
    ),
)
parser.add_argument(
    "--communication_range",
    type=float,
    default=0.025,
    help=(
        "Percentage of overall environment over which agents can communicate "
        "their opinions (default=0.025) "
    ),
)
parser.add_argument(
    "--num_steps",
    type=int,
    default=100,
    help=("Number of steps per episode (default=10,000)"),
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help=(
        "Number of training episodes with "
        "n steps (num_steps) per episode (default=100) "
        "Note that the environment resets every episode"
    ),
)
parser.add_argument(
    "--data_directory_name",
    type=str,
    help=("The data directory name for the current experiment"),
)
parser.add_argument(
    "--training_testing_module_name",
    type=str,
    help=(
        "Name for type of experiment to conduct based on the"
        "experiments developed in the training and testing module"
    ),
)
args = parser.parse_args()

current_time = time.asctime().strip().replace(" ", "_")

current_experiment_data_directory = (
    f"{EXPERIMENT_DATA_ROOT_DIRECTORY}/{args.data_directory_name}_{current_time}"
)

if not os.path.exists(current_experiment_data_directory):
    os.makedirs(current_experiment_data_directory)

file_handler(
    path=(
        f"{current_experiment_data_directory}/experiment_hyperparameters_settings.txt"
    ),
    mode="w",
    func=lambda f: f.write(json.dumps(vars(args), indent=4)),
)


def main(args: argparse.Namespace):
    func = experiment_modules_map[args.training_testing_module_name]
    func(args, data_directory=current_experiment_data_directory)


if __name__ == "__main__":
    main(args=args)
