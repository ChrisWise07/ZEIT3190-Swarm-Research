import gym
from gym import spaces
import numpy as np
import environment_agent_modules


class TiledEnvForNavigation(gym.Env):
    """Custom Environment that follows gym interface"""

    def __init__(self, arg1, arg2):
        super(TiledEnvForNavigation, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=4, shape=(3,), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        self.done = False
        self.tile_grid = environment_agent_modules.create_tile_grid()
        return observation  # reward, done, info can't be included
