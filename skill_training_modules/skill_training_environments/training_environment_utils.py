from typing import List, Tuple
import numpy as np
import random

from environment_agent_modules import (
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
)


environment_type_list = [
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
]

EPSILON = 1e-10


def inverse_sigmoid_for_weighting(broadcast_percentage: float) -> float:
    import numpy as np

    return 100 / ((1 + np.exp(0.1 * (broadcast_percentage - 50))) + EPSILON)


def sigmoid_for_weighting(broadcast_percentage: float) -> float:
    import numpy as np

    return 100 / ((1 + np.exp(-0.1 * (broadcast_percentage - 50))) + EPSILON)


def return_environment_based_on_weighting_list(
    weighting_list: List[int],
    environment_width: int,
    environment_height: int,
) -> Tuple[int, np.ndarray]:
    """
    Returns the environment based on the weighting list.
    """
    from scipy.stats import truncnorm

    index_of_environment = random.choices(
        [0, 1, 2],
        weights=weighting_list,
        k=1,
    ).pop(0)

    return (
        index_of_environment,
        environment_type_list[index_of_environment](
            width=environment_width,
            height=environment_height,
            ratio_of_white_to_black_tiles=truncnorm.rvs(
                (0 - 0.5) / 1, (1 - 0.5) / 1, 0.5, 1
            ),
        ),
    )
