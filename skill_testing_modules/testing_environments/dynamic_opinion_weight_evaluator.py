import os
import sys
import wandb
import numpy as np

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map


class DynamicOpinionWeightEvaluator:
    def __init__(
        self,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        environment_type_name: str,
        ratio_of_white_to_black_tiles: float,
        eval_model_name: str,
        max_num_of_steps: int,
        collective_opinion_weighting: float,
        **kwargs
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = eval_model_name
        self.max_num_of_steps = max_num_of_steps
        self.collective_opinion_weighting = collective_opinion_weighting

    def step(self):
        for agent in self.swarm_agents:
            agent.set_dynamic_opinion_weights()
            agent.decide_to_sense_or_broadcast()
            agent.navigate_and_recieve_opinions(self.tile_grid)

        self.number_of_steps += 1

        if self.number_of_steps == self.max_num_of_steps:
            correct_opinion_weight = np.array(
                agent.new_opinion_weights[self.correct_opinion]
                for agent in self.swarm_agents
            )

            incorrect_opinion_weights = np.array(
                agent.new_opinion_weights[(self.correct_opinion + 1) % 2]
                for agent in self.swarm_agents
            )

            calculated_collective_opinions = np.array(
                agent.calculated_collective_opinion for agent in self.swarm_agents
            )

            wandb.log(
                {
                    "average_distance_from_optimal_weighting": np.mean(
                        (self.opinion_weight_max - correct_opinion_weight)
                        + incorrect_opinion_weights
                    ),
                    "average_collective_opinion_distance_from_correct_opinion": np.mean(
                        abs(self.correct_opinion - calculated_collective_opinions)
                    ),
                }
            )
            return True

    def reset(self):
        self.number_of_steps = 0

        self.tile_grid = self.environment_type(
            width=self.width,
            height=self.height,
            ratio_of_white_to_black_tiles=self.ratio_of_white_to_black_tiles,
        )

        self.correct_opinion = round(
            return_ratio_of_white_to_black_tiles(self.tile_grid)
        )

        all_possible_tiles = []

        for column in range(20):
            for row in range(20):
                all_possible_tiles.append((row, column))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                model_names={
                    "nav_model": "multi_agent_nav",
                    "sense_model": "sense_broadcast_model",
                    "commit_to_opinion_model": (
                        "commit_to_opinion_trainer_DQN_random_agent_False_increase_no_action_punishment"
                    ),
                    "dynamic_opinion_weight_model": self.eval_model_name,
                },
                current_direction_facing=1,
                needs_models_loaded=True,
                collective_opinion_weighting=self.collective_opinion_weighting,
            )
            for _ in range(self.num_of_swarm_agents)
        ]
