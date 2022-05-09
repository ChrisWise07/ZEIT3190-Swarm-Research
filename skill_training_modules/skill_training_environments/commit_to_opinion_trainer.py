import os
import sys
from tkinter import SW
from typing import List
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


class CommitToOpinionTrainer(gym.Env):
    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        **kwargs,
    ):
        super(CommitToOpinionTrainer, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(width * height), shape=(4,), dtype=np.float32
        )
        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type_weighting = [33, 33, 33]
        self.model = None

    def set_model(self, model: PPO):
        self.model = model

    def step(self, action):
        agent_opinion = self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].calculate_opinion()

        for pos, agent in enumerate(self.swarm_agents):
            if not agent.committed_to_opinion:
                if pos == self.position_of_swarm_agent_to_train:
                    agent.committed_to_opinion = action
                else:
                    if self.model:
                        agent.committed_to_opinion = self.model.predict(
                            agent.return_commit_decision_states()
                        )[0].item()
                    else:
                        agent.committed_to_opinion = random.randint(0, 1)

                if agent.committed_to_opinion:
                    self.committed_agents_count += 1

            agent.perform_decision_navigate_opinion_update_cycle(
                tile_grid=self.tile_grid
            )

        agent_is_committed_to_opinion = self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].committed_to_opinion

        if agent_is_committed_to_opinion:
            if agent_opinion == self.correct_opinion:
                self.correct_commitment_count += 1
                reward = 100
            else:
                reward = -200
        else:
            reward = -0.05

        self.num_steps += 1

        if (
            self.num_steps == self.max_num_steps
            or self.committed_agents_count == self.num_of_swarm_agents
        ):
            self.done = True

            correct_commitment_ratio = (
                self.correct_commitment_count / self.num_of_swarm_agents
            )

            self.environment_type_weighting[self.index_of_environment] = round(
                inverse_sigmoid_for_weighting(correct_commitment_ratio * 100)
            )

            wandb.log(
                {
                    "correct_commitment_ratio": correct_commitment_ratio,
                    "steps taken": self.num_steps,
                }
            )

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
        self.committed_agents_count = 0
        self.correct_commitment_count = 0

        self.index_of_environment = random.choices(
            [0, 1, 2],
            weights=self.environment_type_weighting,
            k=1,
        ).pop(0)

        self.tile_grid = environment_type_list[self.index_of_environment](
            width=self.width,
            height=self.height,
            ratio_of_white_to_black_tiles=stats.truncnorm.rvs(
                (0 - 0.5) / 1, (1 - 0.5) / 1, 0.5, 1
            ),
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

        self.position_of_swarm_agent_to_train = random.randint(
            0, self.num_of_swarm_agents - 1
        )

        return self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].return_sense_broadcast_states()
