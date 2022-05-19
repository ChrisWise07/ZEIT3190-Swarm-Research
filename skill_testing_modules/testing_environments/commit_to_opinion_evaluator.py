import os
import sys
import numpy as np
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map


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
        **kwargs
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.max_num_of_steps = max_num_of_steps
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = eval_model_name

    def step(self):
        self.total_num_of_steps += 1
        for agent in self.swarm_agents:
            agent.perform_decision_navigate_opinion_update_cycle(
                tile_grid=self.tile_grid
            )
            if agent.committed_to_opinion:
                self.agents_committed += 1

        if self.agents_committed == self.num_of_swarm_agents:
            correct_commitments_count = 0

            for agent in self.swarm_agents:
                if agent.calculate_opinion() == self.correct_opinion:
                    correct_commitments_count += 1

            wandb.log(
                {
                    "time_taken": self.total_num_of_steps / 60,
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

        self.total_num_of_steps = 0

        all_possible_tiles = []

        for column in range(20):
            for row in range(20):
                all_possible_tiles.append((row, column))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                model_names={
                    "nav_model": "multi_agent_nav",
                    "sense_model": "obvs_based_reward_follow_agent_sense_broad",
                    "commit_to_opinion_model": self.eval_model_name,
                },
                current_direction_facing=1,
                needs_models_loaded=True,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.agents_committed = 0
        self.time_taken = 0
        self.percentage_with_correct_opinions = 0.0
