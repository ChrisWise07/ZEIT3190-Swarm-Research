import os
import sys
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import wandb
import random
import scipy.stats as stats

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)

from .training_environment_utils import (
    environment_type_list,
    inverse_sigmoid_for_weighting,
)


class SenseBroadcastTrainer(gym.Env):
    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        **kwargs,
    ):
        super(SenseBroadcastTrainer, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(width * height), shape=(4,), dtype=np.float32
        )
        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type_weighting = [33, 33, 33]
        (
            self.previous_true_positives,
            self.previous_false_positives,
            self.previous_true_negatives,
            self.previous_false_negatives,
        ) = (0, 0, 0, 0)

    def set_model(self, model: PPO):
        self.model = model

    def calculate_reward(self, agent: SwarmAgent) -> int:
        if not agent.sensing:  # agent is broadcasting
            if agent.calculate_opinion() != self.correct_opinion:
                self.broadcast_false_positives += 1
                return -(self.width * self.height) / (1 + agent.num_of_cells_observed)

            self.broadcast_true_positives += 1
            return agent.num_of_cells_observed

        if agent.calculate_opinion() != self.correct_opinion:
            self.broadcast_true_negatives += 1
            return (self.width * self.height) / (1 + agent.num_of_cells_observed)

        self.broadcast_false_negatives += 1
        return -agent.num_of_cells_observed

    def calculate_broadcast_accuracy(self) -> float:
        return (self.broadcast_true_positives + self.broadcast_true_negatives) / (
            self.broadcast_true_positives
            + self.broadcast_false_positives
            + self.broadcast_true_negatives
            + self.broadcast_false_negatives
        )

    def step(self, action):
        for agent in self.swarm_agents:
            if agent != self.agent_to_train:
                agent.sensing = self.model.predict(
                    agent.return_sense_broadcast_states()
                )[0].item()
            else:
                agent.sensing = action
                self.reward = self.calculate_reward(agent)

            agent.navigate(tile_grid=self.tile_grid)
            agent.recieve_local_opinions(tile_grid=self.tile_grid)

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

        # self.agent_to_train = random.choice(self.swarm_agents)

        return (
            self.agent_to_train.return_sense_broadcast_states(),
            self.reward,
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.num_steps = 0
        self.decision_correctness_tracker = np.zeros(self.max_num_steps)

        self.index_of_environment = random.choices(
            [0, 1, 2],
            weights=self.environment_type_weighting,
            k=1,
        ).pop(0)

        self.tile_grid = environment_type_list[self.index_of_environment](
            self.width,
            self.height,
            stats.truncnorm.rvs((0 - 0.5) / 1, (1 - 0.5) / 1, 0.5, 1),
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

        self.broadcast_true_positives = 0
        self.broadcast_false_positives = 0
        self.broadcast_true_negatives = 0
        self.broadcast_false_negatives = 0

        self.agent_to_train = random.choice(self.swarm_agents)

        return self.agent_to_train.return_sense_broadcast_states()
