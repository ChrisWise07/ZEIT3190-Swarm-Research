import os
import sys
import numpy as np
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import create_nonclustered_tile_grid, SwarmAgent


class CellsPerMinuteEvaluator:
    def __init__(
        self,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        eval_model_name: str,
        max_num_of_steps: int,
        **kwargs
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.eval_model_name = eval_model_name
        self.max_num_of_steps = max_num_of_steps

    def step(self):
        for agent in self.swarm_agents:
            agent.navigate(tile_grid=self.tile_grid)

        self.num_of_steps += 1

        if (self.num_of_steps % 60) == 0:
            for pos, agent in enumerate(self.swarm_agents):
                current_num_of_cells_visited = agent.return_num_of_cells_visited()

                self.num_of_cells_visited_by_agent_in_minute[pos] = (
                    current_num_of_cells_visited
                    - self.num_of_cells_visited_by_agent[pos]
                )

                self.num_of_cells_visited_by_agent[pos] = current_num_of_cells_visited

            wandb.log(
                {
                    "average_num_of_new_cells_visited_in_minute": np.mean(
                        self.num_of_cells_visited_by_agent_in_minute
                    )
                }
            )

        if self.num_of_steps == self.max_num_of_steps:
            total_number_of_cells_visited = 0

            for agent in self.swarm_agents:
                total_number_of_cells_visited += agent.return_num_of_cells_visited()

            wandb.log(
                {
                    "average_total_num_of_cells_visited": total_number_of_cells_visited
                    / self.num_of_swarm_agents
                }
            )

    def reset(self):
        self.tile_grid = create_nonclustered_tile_grid(
            width=self.width, height=self.height
        )

        self.num_of_steps = 0

        all_possible_tiles = []

        for column in range(self.width):
            for row in range(self.height):
                all_possible_tiles.append((row, column))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                current_direction_facing=1,
                needs_models_loaded=True,
                model_names={
                    "nav_model": self.eval_model_name,
                    "sense_model": "sense_broadcast_model",
                },
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.num_of_cells_visited_by_agent = np.zeros(shape=(self.num_of_swarm_agents,))

        self.num_of_cells_visited_by_agent_in_minute = np.zeros(
            shape=(self.num_of_swarm_agents,)
        )
