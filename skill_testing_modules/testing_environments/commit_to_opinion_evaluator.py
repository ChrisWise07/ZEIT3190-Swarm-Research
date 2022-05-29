import os
import sys
import wandb
import numpy as np

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)
SECONDS_IN_MINUTE = 60

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map
from helper_files.utils import (
    return_list_of_coordinates_column_by_columns,
)


class CommitToOpinionEvaluator:
    def __init__(
        self,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        max_num_of_steps: int,
        environment_type_name: str,
        ratio_of_white_to_black_tiles: float,
        eval_model_name: str,
        **kwargs,
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.max_num_of_steps = max_num_of_steps
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = eval_model_name

    def set_time_to_first_commit(self, pos: int, time: int):
        self.time_to_first_commit[pos] = time

    def step(self):
        for pos, agent in enumerate(self.swarm_agents):
            print(f"Agent {pos}'s commitment: {agent.committed_to_opinion}")
            if not agent.committed_to_opinion:
                print(f"Agent {pos}'s is deciding if it wants to commit")
                agent.decide_if_to_commit()
                if agent.committed_to_opinion:
                    print(f"Agent {pos}'s decided to commit to opinion")
                    self.agents_committed += 1
                    self.set_time_to_first_commit(
                        pos, self.num_steps / SECONDS_IN_MINUTE
                    )

            agent.perform_decision_navigate_opinion_update_cycle(
                tile_grid=self.tile_grid
            )

        self.num_steps += 1

        if self.agents_committed == self.num_of_swarm_agents:
            correct_commitments_count = 0

            for agent in self.swarm_agents:
                if agent.calculate_opinion() == self.correct_opinion:
                    correct_commitments_count += 1
            print(f"Correct commitments: {correct_commitments_count}")
            print(f"self.time_to_first_commit: {self.time_to_first_commit}")

            wandb.log(
                {
                    "time_taken": np.average(self.time_to_first_commit),
                    "percentage_of_correctly_commited_agents": (
                        correct_commitments_count / self.num_of_swarm_agents
                    ),
                }
            )
            return True

    def reset(self):
        self.tile_grid = self.environment_type(
            width=self.width,
            height=self.height,
            ratio_of_white_to_black_tiles=self.ratio_of_white_to_black_tiles,
        )

        self.correct_opinion = round(
            return_ratio_of_white_to_black_tiles(self.tile_grid)
        )

        self.num_steps = 0

        list_of_coordinates_to_distribute_agents_over = (
            return_list_of_coordinates_column_by_columns(
                num_of_columns=self.width, num_of_rows=self.height
            )
        )

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(
                    self.tile_grid[list_of_coordinates_to_distribute_agents_over.pop(0)]
                ),
                needs_models_loaded=True,
                model_names={
                    "nav_model": "multi_agent_nav_model",
                    "sense_model": "sense_broadcast_model",
                    "commit_to_opinion_model": self.eval_model_name,
                },
                current_direction_facing=1,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.agents_committed = 0
        self.percentage_with_correct_opinions = 0.0
        self.time_to_first_commit = np.zeros(self.num_of_swarm_agents)
