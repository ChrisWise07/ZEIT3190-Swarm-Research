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
