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
    sigmoid_for_weighting,
    inverse_sigmoid_for_weighting,
    EPSILON,
)


class DynamicOpinionWeightingTrainer(gym.Env):
    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        random_agent_per_step: bool,
        **kwargs,
    ):
        super(DynamicOpinionWeightingTrainer, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )
        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type_weighting = [33, 33, 33]
        self.random_agent_per_step = random_agent_per_step
        self.model = None

    def set_model(self, model: PPO):
        self.model = model

    def calculate_reward(self, agent: SwarmAgent) -> int:
        self.correct_opinion_weighting = agent.opinion_weights[self.correct_opinion]
        self.incorrect_opinion_weighting = agent.opinion_weights[
            (self.correct_opinion + 1) % 2
        ]
        return (1 / (1.01 - self.correct_opinion_weighting)) - (
            100 * self.incorrect_opinion_weighting
        )

    def step(self, action):
        for agent in self.swarm_agents:
            if agent != self.agent_to_train:
                if self.model is None:
                    agent.opinion_weights = [0.9, 0.9]
                else:
                    agent.opinion_weights = self.model.predict(
                        agent.return_opinion_weight_states()
                    )[0] 
            else:
                agent.opinion_weights = [((weight + 1) / 2) for weight in action] #[0.0 - 1.0, 0.0 - 1.0]
                reward = self.calculate_reward(agent)

            agent.perform_decision_navigate_opinion_update_cycle(
                tile_grid=self.tile_grid
            )

        wandb.log(
            {
                "correct_opinion_weighting": self.correct_opinion_weighting,
                "incorrect_opinion_weighting": self.incorrect_opinion_weighting,
            }
        )

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True

            self.environment_type_weighting[self.index_of_environment] = round(
                (
                    inverse_sigmoid_for_weighting(self.correct_opinion_weighting * 100)
                    + sigmoid_for_weighting(self.incorrect_opinion_weighting * 100)
                )
                / 2
            )

        if self.random_agent_per_step:
            self.agent_to_train = random.choice(self.swarm_agents)

        return (
            self.agent_to_train.return_opinion_weight_states(),
            reward,
            self.done,
            {},
        )

    def reset(self):
        self.done = False
        self.num_steps = 0

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

        for column in range(20):
            for row in range(20):
                all_possible_tiles.append((row, column))

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(self.tile_grid[all_possible_tiles.pop(0)]),
                current_direction_facing=random.randint(0, 3),
                needs_models_loaded=True,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.agent_to_train = random.choice(self.swarm_agents)

        return self.agent_to_train.return_opinion_weight_states()
