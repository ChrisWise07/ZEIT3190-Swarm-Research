import os
import sys
import numpy as np
import gym
from gym import spaces
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
    SwarmAgent,
    return_ratio_of_white_to_black_tiles,
)


class SenseBroadcastTrainer(gym.Env):
    def __init__(
        self,
        max_num_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        communication_range: int,
        **kwargs
    ):
        super(SenseBroadcastTrainer, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(width * height), shape=(4,), dtype=np.float32
        )
        self.max_num_steps = max_num_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.communication_range = communication_range

    def step(self, action):
        self.num_steps += 1

        self.swarm_agents[0].sensing = bool(action)
        self.swarm_agents[0].navigate(tile_grid=self.tile_grid)
        self.swarm_agents[0].recieve_local_opinions(tile_grid=self.tile_grid)

        for agent in self.swarm_agents[1:]:
            agent.sensing = bool(random.randint(0, 1))
            agent.navigate(tile_grid=self.tile_grid)
            agent.recieve_local_opinions(tile_grid=self.tile_grid)

        agent_opinion = self.swarm_agents[0].return_opinion()

        if agent_opinion is None:
            if self.swarm_agents[0].calculate_opinion() != self.correct_opinion:
                reward = 1
                self.decision_correctness_tracker[self.num_steps - 1] = 1
            else:
                reward = -1
        else:
            if agent_opinion == self.correct_opinion:
                self.decision_correctness_tracker[self.num_steps - 1] = 1
                reward = 2
            else:
                reward = -2

        if self.num_steps == self.max_num_steps:
            self.done = True
            wandb.log(
                {
                    "percentage of episode correctly sensing/broadcasting": np.mean(
                        self.decision_correctness_tracker
                    )
                }
            )

        return (
            self.swarm_agents[0].return_sense_broadcast_states(),
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

        return self.swarm_agents[0].return_sense_broadcast_states()
