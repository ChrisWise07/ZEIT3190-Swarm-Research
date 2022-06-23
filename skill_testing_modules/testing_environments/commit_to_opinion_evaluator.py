import os
import random
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
        commitment_threshold: float,
        **kwargs,
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.max_num_of_steps = max_num_of_steps
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = None
        self.commitment_threshold = commitment_threshold

    def set_time_to_first_commit(self, pos: int, time: int):
        self.time_to_first_commit[pos] = time

    def make_commitment_decision(self, agent: SwarmAgent):
        if self.eval_model_name is not None:
            agent.decide_if_to_commit()
        else:
            # if (
            #     agent.calculated_collective_opinion < self.commitment_threshold
            #     or (1 - agent.calculated_collective_opinion) < self.commitment_threshold
            # ):
            #     agent.committed_to_opinion = True
            agent.committed_to_opinion = random.choices([0, 1], weights=[95, 5]).pop(0)

    def step(self):
        # for pos, agent in enumerate(self.swarm_agents):
        #     if agent.return_ratio_of_total_environment_cells_observed() > (
        #         1 / self.height
        #     ):
        #         if not agent.committed_to_opinion:
        #             self.make_commitment_decision(agent)
        #             if agent.committed_to_opinion:
        #                 self.agents_committed += 1
        #                 self.set_time_to_first_commit(pos, self.num_steps)

        #         agent.perform_decision_navigate_opinion_update_cycle(
        #             tile_grid=self.tile_grid
        #         )
        #     else:
        #         agent.navigate(self.tile_grid)

        for pos, agent in enumerate(self.swarm_agents):
            if not agent.committed_to_opinion:
                self.make_commitment_decision(agent)
                if agent.committed_to_opinion:
                    self.agents_committed += 1
                    self.set_time_to_first_commit(pos, self.num_steps)

            agent.perform_decision_navigate_opinion_update_cycle(
                tile_grid=self.tile_grid
            )

        self.num_steps += 1

        if (
            self.agents_committed == self.num_of_swarm_agents
            or self.num_steps == self.max_num_of_steps
        ):
            correct_commitments_count = 0

            for agent in self.swarm_agents:
                if (
                    round(agent.calculated_collective_opinion) == self.correct_opinion
                    and agent.committed_to_opinion
                ):
                    correct_commitments_count += 1

            wandb.log(
                {
                    "time_taken": np.average(
                        self.time_to_first_commit / SECONDS_IN_MINUTE
                    ),
                    "percentage_of_correctly_commited_agents": (
                        correct_commitments_count / self.num_of_swarm_agents
                    ),
                    "total_number_of_commited_agents": self.agents_committed,
                }
            )

            return True

        return False

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
                    "sense_model": "sense_broadcast_model",
                    "commit_to_opinion_model": self.eval_model_name,
                },
                current_direction_facing=1,
                total_number_of_environment_cells=self.width * self.height,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.agents_committed = 0
        self.time_to_first_commit = np.zeros(self.num_of_swarm_agents)
