import os
import sys
import random
import wandb

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import create_nonclustered_tile_grid, SwarmAgent


class CellsPerMinuteEvaluatorFreeRegion:
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
        if self.eval_model_name:
            self.swarm_agent.model_navigate(tile_grid=self.tile_grid)
        else:
            self.swarm_agent.perform_navigation_action(
                action=random.randint(0, 2), tile_grid=self.tile_grid
            )

        self.step_number += 1

        if (self.step_number % 60) == 0:
            current_num_of_cells_visited = (
                self.swarm_agent.return_num_of_cells_visited()
            )

            cells_visited_in_minute = (
                current_num_of_cells_visited
                - self.swarm_agent_previous_num_of_cells_visited
            )

            self.swarm_agent_previous_num_of_cells_visited = (
                current_num_of_cells_visited
            )

            wandb.log(
                {"average_num_of_new_cells_visited_in_minute": cells_visited_in_minute}
            )

        if self.step_number == self.max_num_of_steps:

            wandb.log(
                {
                    "average_total_num_of_cells_visited": self.swarm_agent.return_num_of_cells_visited()
                }
            )

    def reset(self):
        self.tile_grid = create_nonclustered_tile_grid(
            width=self.width, height=self.height
        )

        self.swarm_agent_previous_num_of_cells_visited = 0

        self.swarm_agent = SwarmAgent(
            starting_cell=(self.tile_grid[(0, 0)]),
            current_direction_facing=1,
            needs_models_loaded=True,
            model_names={"nav_model": self.eval_model_name},
        )

        self.step_number = 0
