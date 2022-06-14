import os
import sys
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .testing_utils import environment_type_map

from helper_files.utils import (
    return_list_of_coordinates_column_by_columns,
)


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
        **kwargs,
    ):
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.max_num_of_steps = max_num_of_steps
        self.environment_type = environment_type_map[environment_type_name]
        self.ratio_of_white_to_black_tiles = ratio_of_white_to_black_tiles
        self.eval_model_name = eval_model_name

    def wrapper_agent_decision_to_sense_or_broadcast(self, agent: SwarmAgent) -> None:
        if self.eval_model_name is not None:
            agent.decide_to_sense_or_broadcast()
        else:
            # chance_of_broadcasting = 0.5 * (
            #     agent.return_ratio_of_total_environment_cells_observed()
            #     + (
            #         1
            #         - abs(
            #             agent.calculate_opinion() - agent.calculated_collective_opinion
            #         )
            #     )
            # )
            # agent.sensing = random.choices(
            #     [0, 1],
            #     weights=[
            #         100 * chance_of_broadcasting,
            #         100 - (100 * chance_of_broadcasting),
            #     ],
            #     k=1,
            # )[0]
            agent.sensing = random.randint(0, 1)

    def step(self):
        (
            broadcast_true_positves,
            broadcast_false_positves,
            broadcast_true_negatives,
            broadcast_false_negatives,
        ) = (0, 0, 0, 0)

        for agent in self.swarm_agents:
            agent_opinion = agent.calculate_opinion()
            self.wrapper_agent_decision_to_sense_or_broadcast(agent)
            agent.navigate_and_recieve_opinions(self.tile_grid)

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
                "percentage_of_correct_opinions_shared": (
                    (
                        broadcast_true_positves
                        / (broadcast_true_positves + broadcast_false_negatives)
                    )
                    if (broadcast_true_positves + broadcast_false_negatives)
                    else 0.0
                ),
                "percentage_of_incorrect_opinions_shared": (
                    (
                        broadcast_false_positves
                        / (broadcast_false_positves + broadcast_true_negatives)
                    )
                    if (broadcast_false_positves + broadcast_true_negatives)
                    else 0.0
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
                    "sense_model": self.eval_model_name,
                },
                current_direction_facing=1,
                total_number_of_environment_cells=self.width * self.height,
            )
            for _ in range(self.num_of_swarm_agents)
        ]
