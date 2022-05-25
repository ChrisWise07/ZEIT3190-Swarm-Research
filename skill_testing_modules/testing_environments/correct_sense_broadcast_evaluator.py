import os
import sys
import wandb

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map


class CorrectSenseBroadcastEvaluator:
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
        (
            broadcast_true_positves,
            broadcast_false_positves,
            broadcast_true_negatives,
            broadcast_false_negatives,
        ) = (0, 0, 0, 0)

        for agent in self.swarm_agents:
            agent_opinion = agent.calculate_opinion()
            agent.decide_to_sense_or_broadcast()
            agent.navigate(self.tile_grid)
            agent.recieve_local_opinions(self.tile_grid)

            if not (agent.sensing):  # broadcasting
                if agent_opinion != self.correct_opinion:
                    broadcast_false_positves += 1
                    continue

                broadcast_true_positves += 1
                continue

            if agent_opinion != self.correct_opinion:
                broadcast_true_negatives += 1
                continue

            broadcast_false_negatives += 1
            continue

        wandb.log(
            {
                "broadcast_accuracy": (
                    broadcast_true_positves + broadcast_true_negatives
                )
                / (
                    broadcast_true_positves
                    + broadcast_false_negatives
                    + broadcast_true_negatives
                    + broadcast_false_positves
                ),
                "broadcast_true_positives": broadcast_true_positves,
                "broadcast_false_positives": broadcast_false_positves,
                "broadcast_true_negatives": broadcast_true_negatives,
                "broadcast_false_negatives": broadcast_false_negatives,
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

        for column in range(20):
            for row in range(20):
                all_possible_tiles.append((row, column))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                model_names={
                    "nav_model": "multi_agent_nav",
                    "sense_model": self.eval_model_name,
                    "commit_to_opinion_model": "commit_to_opinion_model",
                },
                current_direction_facing=1,
                needs_models_loaded=True,
            )
            for _ in range(self.num_of_swarm_agents)
        ]
