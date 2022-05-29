import os
import sys
import gym
import numpy as np
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)
SECONDS_IN_MINUTE = 60

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


class CommitToOpinionTrainer(gym.Env):
    from stable_baselines3 import PPO, DQN
    from typing import Union

    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        random_agent_per_step: bool,
        **kwargs,
    ):
        from gym import spaces
        from numpy import float32

        super(CommitToOpinionTrainer, self).__init__()

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=float(max_num_of_steps), shape=(2,), dtype=float32
        )
        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.environment_type_weighting = [33, 33, 33]
        self.model = None
        self.random_agent_per_step = random_agent_per_step

    def set_model(self, model: Union[PPO, DQN]):
        self.model = model

    def calculate_reward(self, agent: SwarmAgent) -> int:
        if not (agent.committed_to_opinion):
            return -0.01 * agent.num_of_cells_observed

        if agent.calculate_opinion() != self.correct_opinion:
            return -((self.width * self.height) / agent.num_of_cells_observed)

        return agent.num_of_cells_observed

    def return_action_for_other_agent(self, agent: SwarmAgent):
        if self.model is not None:
            return self.model.predict(agent.return_commit_decision_states())[0].item()
        return random.choice([0, 1])

    def set_time_to_first_commit(self, pos: int, time: int):
        self.time_to_first_commit[pos] = time

    def step(self, action):
        for pos, agent in enumerate(self.swarm_agents):
            if agent != self.agent_to_train:
                agent.committed_to_opinion = self.return_action_for_other_agent(agent)
            else:
                agent.committed_to_opinion = action
                reward = self.calculate_reward(agent)

            if agent.committed_to_opinion and not self.time_to_first_commit[pos]:
                self.set_time_to_first_commit(pos, self.num_steps)

            agent.perform_decision_navigate_opinion_update_cycle(self.tile_grid)

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True

            correct_commitment_count = 0

            for agent in self.swarm_agents:
                if agent.calculate_opinion() == self.correct_opinion:
                    correct_commitment_count += 1

            correct_commitment_ratio = (
                correct_commitment_count / self.num_of_swarm_agents
            )

            wandb.log(
                {
                    "percentage_of_correct_commitments": correct_commitment_ratio,
                    "average_commitment_time": np.average(
                        self.time_to_first_commit / SECONDS_IN_MINUTE
                    ),
                }
            )

            self.environment_type_weighting[self.index_of_environment] = round(
                (inverse_sigmoid_for_weighting(correct_commitment_ratio * 100))
            )

        if self.random_agent_per_step:
            self.agent_to_train = random.choice(self.swarm_agents)

        return (
            self.agent_to_train.return_commit_decision_states(),
            reward,
            self.done,
            {},
        )

    def reset(self):
        from scipy.stats import truncnorm

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
            ratio_of_white_to_black_tiles=truncnorm.rvs(
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
                needs_models_loaded=True,
                current_direction_facing=random.randint(0, 3),
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.agent_to_train = random.choice(self.swarm_agents)
        self.time_to_first_commit = np.zeros(self.num_of_swarm_agents)

        return self.agent_to_train.return_commit_decision_states()
