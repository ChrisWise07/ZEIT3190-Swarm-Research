import os
import sys
import numpy as np
import gym
from gym import spaces
import wandb
import random
from stable_baselines3 import PPO

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    create_nonclustered_tile_grid,
    SwarmAgent,
)


class MultiAgentNavigationTrainer(gym.Env):
    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        **kwargs
    ):
        super(MultiAgentNavigationTrainer, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=0, high=4, shape=(2,), dtype=int)
        self.max_num_of_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents

    def set_model(self, model: PPO):
        self.model = model

    def step(self, action):
        self.num_steps += 1

        for pos, agent in enumerate(self.swarm_agents):
            if pos == self.position_of_swarm_agent_to_train:
                agent.perform_navigation_action(action=action, tile_grid=self.tile_grid)
            else:
                agent.perform_navigation_action(
                    action=self.model.predict(
                        agent.get_navigation_states(tile_grid=self.tile_grid)
                    )[0].item(),
                    tile_grid=self.tile_grid,
                )

        reward = self.swarm_agents[
            self.position_of_swarm_agent_to_train
        ].return_navigation_reward()

        if (self.num_steps % 60) == 0:
            for pos, agent in enumerate(self.swarm_agents):
                current_num_of_cells_visited = agent.return_num_of_cells_visited()

                self.num_of_cells_visited_by_agent_in_minute[pos] = (
                    current_num_of_cells_visited
                    - self.num_of_cells_visited_by_agent[pos]
                )

                self.num_of_cells_visited_by_agent[pos] = current_num_of_cells_visited

            wandb.log(
                {
                    "average_num_of_new_cells_visited_in_minute": np.mean(
                        self.num_of_cells_visited_by_agent_in_minute
                    )
                }
            )

            wandb.log(
                {
                    "average_total_num_of_cells_visited": np.mean(
                        self.num_of_cells_visited_by_agent
                    )
                }
            )

        if self.num_steps == self.max_num_of_steps:
            for pos, agent in enumerate(self.swarm_agents):
                current_num_of_cells_visited = agent.return_num_of_cells_visited()
                self.num_of_cells_visited_by_agent[pos] = current_num_of_cells_visited

            wandb.log(
                {
                    "average_total_num_of_cells_visited": np.mean(
                        self.num_of_cells_visited_by_agent
                    )
                }
            )

            self.done = True

        return (
            np.array(
                self.swarm_agents[
                    self.position_of_swarm_agent_to_train
                ].get_navigation_states(self.tile_grid)
            ),
            reward,
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

        self.num_of_cells_visited_by_agent = np.zeros(shape=(self.num_of_swarm_agents,))

        self.num_of_cells_visited_by_agent_in_minute = np.zeros(
            shape=(self.num_of_swarm_agents,)
        )

        self.position_of_swarm_agent_to_train = random.randint(
            0, self.num_of_swarm_agents - 1
        )

        return np.array(
            self.swarm_agents[
                self.position_of_swarm_agent_to_train
            ].get_navigation_states(self.tile_grid)
        )
