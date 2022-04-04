from typing import Dict
import numpy as np


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
