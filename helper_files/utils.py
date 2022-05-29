from typing import Dict, List, Tuple, Union
import numpy as np
from stable_baselines3 import PPO, DQN


def return_statistical_analysis_data(data: np.ndarray) -> Dict:
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "min": np.min(data),
        "max": np.max(data),
        "standard_deviation": np.std(data),
    }


def file_handler(path, mode, func):
    try:
        with open(path, mode) as f:
            return func(f)
    except FileNotFoundError:
        return 0


def calculate_optimal_number_of_steps_needed(
    tiled_environment_width: int,
    tiled_environment_height: int,
) -> int:
    """
    Calculates the optimal number of steps needed to complete the task.
    """

    return int(
        (
            (tiled_environment_height * tiled_environment_width)
            - 2
            + (2 * tiled_environment_width)
        )
    )


def return_list_of_coordinates_column_by_columns(
    num_of_columns: int, num_of_rows: int
) -> List[Tuple[int, int]]:
    """
    Returns a list of coordinates going by row then columns.
    """

    return [
        (row, column) for column in range(num_of_columns) for row in range(num_of_rows)
    ]


def return_model(model_name: str) -> Union[PPO, DQN]:
    return {
        "PPO": PPO,
        "DQN": DQN,
    }[model_name]
