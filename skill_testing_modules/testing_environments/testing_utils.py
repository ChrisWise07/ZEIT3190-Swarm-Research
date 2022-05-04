from environment_agent_modules import (
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
)

environment_type_map = {
    "nonclustered_tile_grid": create_nonclustered_tile_grid,
    "clustered_inital_observation_useful_tile_grid": create_clustered_inital_observation_useful_tile_grid,
    "clustered_inital_observation_not_useful_tile_grid": create_clustered_inital_observation_not_useful_tile_grid,
}
