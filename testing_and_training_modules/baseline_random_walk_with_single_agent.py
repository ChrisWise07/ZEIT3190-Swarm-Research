import numpy as np
import json
import argparse
from random import choice
from environment_agent_modules import tiled_environment
from environment_agent_modules import create_nonclustered_tile_grid, SwarmAgent
from .utils import return_statistical_analysis_data, file_handler


def perform_random_walk_and_record_num_cells_visited(
    width: int,
    height: int,
    ratio_of_white_to_black_tiles: float,
    num_episodes: int,
    num_steps: int,
) -> np.ndarray:
    num_of_unique_cells_visited = np.empty(shape=(num_episodes,))

    for episode in range(num_episodes):
        tiled_environment = create_nonclustered_tile_grid(
            width=width,
            height=height,
            ratio_of_white_to_black_tiles=ratio_of_white_to_black_tiles,
        )
        swarm_agent = SwarmAgent(id=1, starting_cell=tiled_environment[(0, 0)])

        for step in range(num_steps):
            swarm_agent.perform_navigation_action(
                action=choice(range(0, 3)), tile_grid=tiled_environment
            )

        num_of_unique_cells_visited[episode] = len(swarm_agent.cells_visited)

    return num_of_unique_cells_visited


def baseline_random_walk_with_single_agent(
    args: argparse.Namespace, data_directory: str
) -> None:
    num_of_unique_cells_visited = perform_random_walk_and_record_num_cells_visited(
        width=args.tiled_environment_width,
        height=args.tiled_environment_height,
        ratio_of_white_to_black_tiles=args.ratio_of_white_to_black_tiles,
        num_episodes=args.num_episodes,
        num_steps=args.num_steps,
    )
    stats_data = return_statistical_analysis_data(num_of_unique_cells_visited)

    file_handler(
        path=(f"{data_directory}/stats_data_on_num_of_unique_cells_visited.txt"),
        mode="w",
        func=lambda f: f.write(json.dumps(stats_data, indent=4)),
    )
