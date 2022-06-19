import os
import sys
import wandb
import numpy as np

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    MaliciousAgent,
    return_ratio_of_white_to_black_tiles,
)

from helper_files.utils import (
    return_list_of_coordinates_column_by_columns,
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
        max_new_opinion_weighting: float,
        opinion_weighting_method: str,
        num_of_malicious_agents: int,
        sensing_noise: float,
        communication_noise: float,
        **kwargs,
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = None
        self.max_new_opinion_weighting = max_new_opinion_weighting
        self.opinion_weighting_method = opinion_weighting_method
        self.num_of_malicious_agents = num_of_malicious_agents
        self.sensing_noise = sensing_noise
        self.communication_noise = communication_noise

    def update_opinion_weight(self, agent: SwarmAgent):
        if self.eval_model_name:
            agent.set_dynamic_opinion_weights()
        elif self.opinion_weighting_method == "equation_based":
            agent.opinion_weights = [
                agent.return_opinion_weight_based_on_equation(1),
                agent.return_opinion_weight_based_on_equation(0),
            ]

    def step(self):
        for agent in self.malicious_agents:
            agent.navigate(self.tile_grid)

        for agent in self.swarm_agents:
            # self.update_opinion_weight(agent)
            agent.decide_to_sense_or_broadcast()
            agent.navigate_and_recieve_opinions(self.tile_grid)

        correct_opinion_weight = np.array(
            [
                (agent.return_opinion_weight_based_on_equation(self.correct_opinion))
                for agent in self.swarm_agents
            ]
        )

        incorrect_opinion_weights = np.array(
            [
                (
                    agent.return_opinion_weight_based_on_equation(
                        (self.correct_opinion + 1) % 2
                    )
                )
                for agent in self.swarm_agents
            ]
        )

        calculated_collective_opinions = np.array(
            [agent.calculated_collective_opinion for agent in self.swarm_agents]
        )

        wandb.log(
            {
                "average_distance_from_optimal_weighting": np.mean(
                    (self.max_new_opinion_weighting - correct_opinion_weight)
                    + incorrect_opinion_weights
                ),
                "average_collective_opinion_distance_from_correct_opinion": np.mean(
                    abs(self.correct_opinion - calculated_collective_opinions)
                ),
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
                    "dynamic_opinion_model": self.eval_model_name,
                },
                current_direction_facing=1,
                max_new_opinion_weighting=self.max_new_opinion_weighting,
                total_number_of_environment_cells=self.width * self.height,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.malicious_agents = [
            MaliciousAgent(
                starting_cell=(
                    self.tile_grid[list_of_coordinates_to_distribute_agents_over.pop(0)]
                ),
                malicious_opinion=((self.correct_opinion + 1) % 2),
                current_direction_facing=1,
            )
            for _ in range(self.num_of_malicious_agents)
        ]
