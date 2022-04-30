import os
import sys
import numpy as np
import gym
from gym import spaces
from stable_baselines3 import PPO
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
    return_ratio_of_white_to_black_tiles,
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

    def set_model(self, model: PPO):
        self.model = model

    def step(self, action):
        agent_opinion = self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].return_opinion()

        agent_calculated_opinion = self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].calculate_opinion()

        for pos, agent in enumerate(self.swarm_agents):
            agent.sensing = (
                action
                if pos == self.position_of_swarm_agent_to_train
                else self.model.predict(agent.return_sense_broadcast_states())[0].item()
            )
            agent.navigate(tile_grid=self.tile_grid)
            agent.recieve_local_opinions(tile_grid=self.tile_grid)

        if agent_opinion is None:
            if agent_calculated_opinion != self.correct_opinion:
                self.broadcast_true_negatives += 1
                reward = 1
            else:
                self.broadcast_false_negatives += 1
                reward = -1
        else:
            if agent_opinion == self.correct_opinion:
                self.broadcast_true_positves += 1
                reward = 2
            else:
                self.broadcast_false_positves += 1
                reward = -4

        wandb.log(
            {
                "broadcast_accuracy": (
                    self.broadcast_true_positves + self.broadcast_true_negatives
                )
                / (
                    self.broadcast_true_positves
                    + self.broadcast_false_negatives
                    + self.broadcast_true_negatives
                    + self.broadcast_false_positves
                ),
                "broadcast_true_positives": self.broadcast_true_positves,
                "broadcast_false_positives": self.broadcast_false_positves,
                "broadcast_true_negatives": self.broadcast_true_negatives,
                "broadcast_false_negatives": self.broadcast_false_negatives,
            }
        )

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True

        self.position_of_swarm_agent_to_train = random.randint(
            0, self.num_of_swarm_agents - 1
        )

        return (
            self.swarm_agents[
                self.position_of_swarm_agent_to_train
            ].return_sense_broadcast_states(),
            reward,
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.num_steps = 0
        self.max_num_steps = int(self.max_num_steps)
        self.decision_correctness_tracker = np.zeros(self.max_num_steps)

        self.tile_grid = random.choice(
            [
                create_nonclustered_tile_grid,
                create_clustered_inital_observation_useful_tile_grid,
                create_clustered_inital_observation_not_useful_tile_grid,
            ]
        )(self.width, self.height, random.uniform(0.0, 1.0))

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

        self.broadcast_true_positves = 0
        self.broadcast_false_positves = 0
        self.broadcast_true_negatives = 0
        self.broadcast_false_negatives = 0

        self.position_of_swarm_agent_to_train = random.randint(
            0, self.num_of_swarm_agents - 1
        )

        return self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].return_sense_broadcast_states()
