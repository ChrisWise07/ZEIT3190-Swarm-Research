import os
import sys
import gym
import wandb
import random


ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .training_environment_utils import (
    return_environment_based_on_weighting_list,
    inverse_sigmoid_for_weighting,
)

from helper_files import return_list_of_coordinates_column_by_columns


class SenseBroadcastTrainer(gym.Env):
    from stable_baselines3 import PPO, DQN
    from typing import Union

    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        random_agent_per_step: bool,
        **kwargs,
    ):
        super(SenseBroadcastTrainer, self).__init__()

        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.random_agent_per_step = random_agent_per_step

        self.environment_type_weighting = [33, 33, 33]
        self.model = None

        from numpy import float32
        from gym import spaces

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(width * height), shape=(4,), dtype=float32
        )

    def set_model(self, model: Union[PPO, DQN]):
        self.model = model

    def calculate_reward(self, agent: SwarmAgent) -> int:
        if not agent.sensing:  # agent is broadcasting
            if agent.calculate_opinion() != self.correct_opinion:
                self.broadcast_false_positives += 1
                return -(self.width * self.height) / agent.num_of_cells_observed

            self.broadcast_true_positives += 1
            return agent.num_of_cells_observed

        if agent.calculate_opinion() != self.correct_opinion:
            self.broadcast_true_negatives += 1
            return (self.width * self.height) / agent.num_of_cells_observed

        self.broadcast_false_negatives += 1
        return -agent.num_of_cells_observed

    def calculate_broadcast_accuracy(self) -> float:
        return (self.broadcast_true_positives + self.broadcast_true_negatives) / (
            self.broadcast_true_positives
            + self.broadcast_false_positives
            + self.broadcast_true_negatives
            + self.broadcast_false_negatives
        )

    def return_action_for_other_agent(self, agent: SwarmAgent):
        if self.model is not None:
            return self.model.predict(agent.return_sense_broadcast_states())[0].item()
        return random.choice([0, 1])

    def step(self, action):
        for agent in self.swarm_agents:
            if agent != self.agent_to_train:
                agent.sensing = self.return_action_for_other_agent(agent)
            else:
                agent.sensing = action
                reward = self.calculate_reward(agent)

            agent.navigate_and_recieve_opinions(self.tile_grid)

        broadcast_accuracy = self.calculate_broadcast_accuracy()

        wandb.log(
            {
                "broadcast_accuracy": broadcast_accuracy,
                "broadcast_true_positives": self.broadcast_true_positives,
                "broadcast_false_positives": self.broadcast_false_positives,
                "broadcast_true_negatives": self.broadcast_true_negatives,
                "broadcast_false_negatives": self.broadcast_false_negatives,
            }
        )

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True
            self.environment_type_weighting[self.index_of_environment] = round(
                inverse_sigmoid_for_weighting(broadcast_accuracy * 100)
            )

        if self.random_agent_per_step:
            self.agent_to_train = random.choice(self.swarm_agents)

        return (
            self.agent_to_train.return_sense_broadcast_states(),
            reward,
            self.done,
            {},
        )

    def reset(self):
        from numpy import zeros

        self.done = False
        self.num_steps = 0
        self.decision_correctness_tracker = zeros(self.max_num_steps)

        (
            self.index_of_environment,
            self.tile_grid,
        ) = return_environment_based_on_weighting_list(
            self.environment_type_weighting, self.width, self.height
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
                current_direction_facing=random.randint(0, 3),
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.broadcast_true_positives = 0
        self.broadcast_false_positives = 0
        self.broadcast_true_negatives = 0
        self.broadcast_false_negatives = 0

        self.agent_to_train = random.choice(self.swarm_agents)

        return self.agent_to_train.return_sense_broadcast_states()
