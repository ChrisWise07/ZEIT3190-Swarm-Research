import os
import sys
import numpy as np
import gym
from gym import spaces
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import create_nonclustered_tile_grid, SwarmAgent


class SingleAgentNavigationTrainer(gym.Env):
    def __init__(self, max_num_of_steps: int, width: int, height: int, **kwargs):
        super(SingleAgentNavigationTrainer, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=int)
        self.max_num_of_steps = max_num_of_steps
        self.width, self.height = width, height

    def set_model(self, model):
        self.model = model

    def step(self, action):
        self.num_steps += 1

        self.swarm_agent.perform_navigation_action(
            action=action, tile_grid=self.tile_grid
        )

        if (self.num_steps % 60) == 0:
            current_num_of_cells_visited = (
                self.swarm_agent.return_num_of_cells_visited()
            )

            self.num_of_cells_visited_by_agent_in_minute = (
                current_num_of_cells_visited - self.num_of_cells_visited_by_agent
            )

            self.num_of_cells_visited_by_agent = current_num_of_cells_visited

            wandb.log(
                {
                    "average_num_of_new_cells_visited_in_minute": self.num_of_cells_visited_by_agent_in_minute
                }
            )

            wandb.log(
                {
                    "average_total_num_of_cells_visited": self.num_of_cells_visited_by_agent
                }
            )

        if self.num_steps == self.max_num_of_steps:
            current_num_of_cells_visited = (
                self.swarm_agent.return_num_of_cells_visited()
            )

            wandb.log(
                {
                    "average_total_num_of_cells_visited": np.mean(
                        current_num_of_cells_visited
                    )
                }
            )

            self.done = True

        return (
            np.array(self.swarm_agent.get_navigation_states(self.tile_grid)),
            self.swarm_agent.return_navigation_reward(),
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.num_steps = 0
        self.tile_grid = create_nonclustered_tile_grid(
            width=self.width, height=self.height
        )

        self.swarm_agent = SwarmAgent(
            starting_cell=(
                self.tile_grid[
                    (
                        random.randint(0, self.height - 1),
                        random.randint(0, self.width - 1),
                    )
                ]
            ),
            current_direction_facing=random.randint(0, 3),
        )

        self.num_of_cells_visited_by_agent = 0
        self.num_of_cells_visited_by_agent_in_minute = 0

        return np.array(self.swarm_agent.get_navigation_states(self.tile_grid))
