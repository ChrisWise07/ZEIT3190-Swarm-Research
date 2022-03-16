import numpy as np
import gym
from gym import spaces
from environment_agent_modules import create_tile_grid, SwarmAgent, Turn


class TiledEnvForNavigation(gym.Env):
    def __init__(self):
        super(TiledEnvForNavigation, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=4, shape=(3,), dtype=int)

    def step(self, action):
        if action == 0:
            self.swarm_agent.forward_step(tile_grid=self.tile_grid)
        elif action == 1:
            self.swarm_agent.turn(turn_type=Turn.RIGHT)
        elif action == 2:
            self.swarm_agent.turn(turn_type=Turn.LEFT)

        return (
            np.array(self.swarm_agent.get_navigation_states(self.tile_grid)),
            self.swarm_agent.return_navigation_reward(),
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.tile_grid = create_tile_grid(width=5, height=5)
        self.swarm_agent = SwarmAgent(id=1, starting_cell=(self.tile_grid[(0, 0)]))
        return np.array(self.swarm_agent.get_navigation_states(self.tile_grid))
