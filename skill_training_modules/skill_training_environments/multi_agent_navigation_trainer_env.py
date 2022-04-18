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
    SwarmAgent,
)


class MultiAgentNavigationTrainer(gym.Env):
    def __init__(
        self,
        max_num_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        **kwargs
    ):
        super(MultiAgentNavigationTrainer, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=int)
        self.max_num_steps = max_num_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents

    def step(self, action):
        self.num_steps += 1
        self.swarm_agents[0].perform_navigation_action(
            action=action, tile_grid=self.tile_grid
        )
        for agent in self.swarm_agents[1:]:
            agent.navigate(tile_grid=self.tile_grid)

        if self.num_steps == self.max_num_steps:
            self.done = True
            wandb.log({"num_of_cells_visited": len(self.swarm_agents[0].cells_visited)})

        return (
            np.array(self.swarm_agents[0].get_navigation_states(self.tile_grid)),
            self.swarm_agents[0].return_navigation_reward(),
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.num_steps = 0
        self.tile_grid = create_nonclustered_tile_grid(
            width=self.width, height=self.height
        )
        all_possible_tiles = []
        for row in range(self.height):
            for column in range(self.width):
                all_possible_tiles.append((row, column))
        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(
                    self.tile_grid[
                        all_possible_tiles.pop(
                            random.randint(0, len(all_possible_tiles) - 1)
                        )
                    ]
                ),
                current_direction_facing=random.randint(0, 3),
                needs_models_loaded=True,
            )
            for i in range(self.num_of_swarm_agents)
        ]
        return np.array(self.swarm_agents[0].get_navigation_states(self.tile_grid))
