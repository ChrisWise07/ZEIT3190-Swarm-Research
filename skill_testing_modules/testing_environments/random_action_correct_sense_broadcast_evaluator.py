import os
import sys
import numpy as np
from wandb import log
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map


class RandomActionCorrectSenseBroadcastEvaluator:
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

    def step(self, step_number: int):
        for agent in self.swarm_agents:
            agent.sensing = random.randint(0, 1)
            agent.navigate(tile_grid=self.tile_grid)
            agent.recieve_local_opinions(tile_grid=self.tile_grid)

        for agent in self.swarm_agents:
            agent_opinion = agent.return_opinion()

            if agent_opinion is None:
                if agent.calculate_opinion() != self.correct_opinion:
                    self.broadcast_true_negatives[step_number] += 1
                else:
                    self.broadcast_false_negatives[step_number] += 1
            else:
                if agent_opinion == self.correct_opinion:
                    self.broadcast_true_positves[step_number] += 1
                else:
                    self.broadcast_false_positves[step_number] += 1

        log(
            {
                "broadcast_accuracy": (
                    self.broadcast_true_positves[step_number]
                    + self.broadcast_true_negatives[step_number]
                )
                / (
                    self.broadcast_true_positves[step_number]
                    + self.broadcast_false_negatives[step_number]
                    + self.broadcast_true_negatives[step_number]
                    + self.broadcast_false_positves[step_number]
                ),
                "broadcast_true_positives": self.broadcast_true_positves[step_number],
                "broadcast_false_positives": self.broadcast_false_positves[step_number],
                "broadcast_true_negatives": self.broadcast_true_negatives[step_number],
                "broadcast_false_negatives": self.broadcast_false_negatives[
                    step_number
                ],
            }
        )

    def reset(self):
        self.tile_grid = self.environment_type(
            width=self.width,
            height=self.height,
            ratio_of_white_to_black_tiles=self.ratio_of_white_to_black_tiles,
        )

        self.correct_opinion = round(
            return_ratio_of_white_to_black_tiles(self.tile_grid)
        )

        all_possible_tiles = []

        for column in range(self.width):
            for row in range(self.height):
                all_possible_tiles.append((column, row))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                current_direction_facing=random.randint(0, 3),
                needs_models_loaded=True,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.broadcast_true_positves = np.zeros(shape=(self.max_num_of_steps,))
        self.broadcast_false_positves = np.zeros(shape=(self.max_num_of_steps,))
        self.broadcast_true_negatives = np.zeros(shape=(self.max_num_of_steps,))
        self.broadcast_false_negatives = np.zeros(shape=(self.max_num_of_steps,))
