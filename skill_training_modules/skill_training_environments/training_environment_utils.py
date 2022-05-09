import numpy as np

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


def inverse_sigmoid_for_weighting(broadcast_percentage: float) -> float:
    return 100 / ((1 + np.exp(0.1 * (broadcast_percentage - 50))) + 0.00001)
