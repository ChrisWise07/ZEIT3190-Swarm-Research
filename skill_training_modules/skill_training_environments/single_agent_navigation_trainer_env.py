import os
import sys
import numpy as np
import gym
from gym import spaces

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import create_nonclustered_tile_grid, SwarmAgent


class SingleAgentNavigationTrainer(gym.Env):
    def __init__(self, max_steps_num: int):
        super(SingleAgentNavigationTrainer, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=3, shape=(2,), dtype=int)
        self.max_steps = 100

    def step(self, action):
        self.num_steps += 1

        self.swarm_agent.perform_navigation_action(
            action=action, tile_grid=self.tile_grid
        )

        if len(self.swarm_agent.cells_visited) == (
            self.tile_grid.shape[0] * self.tile_grid.shape[1]
        ):
            self.done = True
        if self.num_steps == self.max_steps:
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
        self.tile_grid = create_nonclustered_tile_grid(width=5, height=5)
        self.swarm_agent = SwarmAgent(id=1, starting_cell=(self.tile_grid[(0, 0)]))
        self.observation = np.array(
            self.swarm_agent.get_navigation_states(self.tile_grid)
        )
        return self.observation
