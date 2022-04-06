import argparse
from helper_files import calculate_optimal_number_of_steps_needed
from experiment_modules_map import experiment_modules_map
from skill_training_modules import train_skill_with_environment

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
    default=20,
    help="The height of the tiled environment i.e. the number of rows (default=20)",
)
parser.add_argument(
    "--width",
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
    "--max_num_steps",
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
    default=4.0,
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
    "--testing_module_name",
    type=str,
    default=None,
    help=("The name of the testing module to use (default=None)."),
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

args = parser.parse_args()

if not (args.max_num_steps):
    args.max_num_steps = (
        args.optimal_step_multiplier
        * calculate_optimal_number_of_steps_needed(
            tiled_environment_width=args.width,
            tiled_environment_height=args.height,
        )
    )


def main(args: argparse.Namespace) -> None:
    if args.training_environment:
        args.training_environment = experiment_modules_map[args.training_environment]
        train_skill_with_environment(args=args)
    else:
        experiment_modules_map[args.training_testing_module_name](args)


if __name__ == "__main__":
    main(args=args)
