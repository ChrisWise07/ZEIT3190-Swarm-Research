import argparse
from random import random
from helper_files import calculate_optimal_number_of_steps_needed
from experiment_modules_map import experiment_modules_map
from skill_training_modules import train_skill_with_environment
from skill_testing_modules import test_skill_with_environment

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
    "--height",
    type=int,
    default=25,
    help="The height of the tiled environment i.e. the number of rows (default=20)",
)
parser.add_argument(
    "--width",
    type=int,
    default=25,
    help="The width of the tiled environment i.e. the number of columns (default=20)",
)
parser.add_argument(
    "--num_of_swarm_agents",
    type=int,
    default=12,
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
    type=int,
    default=1,
    help=(
        "Percentage of overall environment over which agents can communicate "
        "their opinions (default=0.025) "
    ),
)
parser.add_argument(
    "--max_num_of_steps",
    type=int,
    default=None,
    help=(
        "Maximum number of steps per episode. "
        "If None, than optimal_step_multiplier "
        "* optimal step number is used (default=None)"
    ),
)
parser.add_argument(
    "--optimal_step_multiplier",
    type=float,
    default=3.0,
    help=(
        "Amount to multiply the optimal step number when "
        "setting the max numnber of steps per episode (default=1.5)"
    ),
)
parser.add_argument(
    "--num_episodes",
    type=int,
    default=100,
    help=(
        "Number of training episodes. "
        "Note that the environment resets every episode (default=100)"
    ),
)
parser.add_argument(
    "--experiment_name",
    type=str,
    help=("The data directory name for the current experiment"),
)
parser.add_argument(
    "--testing_environment",
    type=str,
    default=None,
    help=("The name of the testing module to use (default=None)."),
)
parser.add_argument(
    "--environment_type_name",
    type=str,
    default="nonclustered_tile_grid",
    help=("The name of the environment type (default=nonclustered_tile_grid)."),
)
parser.add_argument(
    "--training_environment",
    type=str,
    default=None,
    help=("The name of the training environment to use (default=None)."),
)
parser.add_argument(
    "--offline",
    type=bool,
    default=False,
    help=("Controls if wandb is ran offline (default=False)"),
)
parser.add_argument(
    "--previous_model",
    type=str,
    default=None,
    help=("Name of the previous model to load (default=None)."),
)
parser.add_argument(
    "--verbose",
    type=int,
    default=1,
    help=("Controls the verbosity of the output (default=1)"),
)
parser.add_argument(
    "--eval_model_name",
    type=str,
    default=None,
    help=("Name of model to evaluate (default=None)"),
)
parser.add_argument(
    "--random_agent_per_step",
    type=str,
    default=False,
    help=("Control if random agent is selected each step (default=None)"),
)
parser.add_argument(
    "--model_type",
    type=str,
    default=None,
    help=("Name of model architecture to train (default=None)"),
)
args = parser.parse_args()


def convert_random_agent_per_step_to_bool(flag: str) -> None:
    return flag in ["True", "true", "1", "yes", "Yes"]


def main(args: argparse.Namespace) -> None:
    """
    Main function for the experiment.

    Args:
        args: The arguments passed to the program.
    """

    args.random_agent_per_step = convert_random_agent_per_step_to_bool(
        args.random_agent_per_step
    )

    if args.training_environment is not None:
        args.training_environment = experiment_modules_map[args.training_environment]
        train_skill_with_environment(args=args)
        return

    args.testing_environment = experiment_modules_map[args.testing_environment]
    test_skill_with_environment(args=args)
    return


if __name__ == "__main__":
    main(args=args)
