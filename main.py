import argparse
import os
import json

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
    "--num_episodes",
    type=int,
    default=100,
    help=("Number of training episodes with 10,000 steps per episode (default=100) "),
)
args = parser.parse_args()

current_experiment_data_directory = (
    f"{ROOT_EXPERIMENT_DATA_DIRECTORY}/{args.data_folder_name}"
)
initial_predictions_images_directory = (
    f"{current_experiment_data_directory}/initial_predictions"
)
final_patches_directory = f"{current_experiment_data_directory}/patches_adv"
final_predictions_images_directory = (
    f"{current_experiment_data_directory}/final_predictions"
)
final_patched_images_directory = (
    f"{current_experiment_data_directory}/final_patched_images"
)
training_data_directory = f"{current_experiment_data_directory}/training_data"
training_loss_printouts_directory = (
    f"{current_experiment_data_directory}/training_loss_printouts"
)
loss_plots_directory = f"{current_experiment_data_directory}/loss_plots_directory"

os.mkdir(current_experiment_data_directory)
os.mkdir(initial_predictions_images_directory)
os.mkdir(final_patches_directory)
os.mkdir(final_predictions_images_directory)
os.mkdir(final_patched_images_directory)
os.mkdir(training_data_directory)
os.mkdir(training_loss_printouts_directory)
os.mkdir(loss_plots_directory)

file_handler(
    path=f"{current_experiment_data_directory}/hyperparameters.txt",
    mode="w",
    func=lambda f: f.write(json.dumps(vars(args), indent=4)),
)
