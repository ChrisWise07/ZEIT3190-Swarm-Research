import os
import sys
import gym
import wandb
import random

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    SwarmAgent,
    MaliciousAgent,
    return_ratio_of_white_to_black_tiles,
)

from .training_environment_utils import (
    sigmoid_for_weighting,
    inverse_sigmoid_for_weighting,
    return_environment_based_on_weighting_list,
)

from helper_files import return_list_of_coordinates_column_by_columns


class DynamicOpinionWeightingTrainer(gym.Env):
    from stable_baselines3 import PPO, DQN
    from typing import Union
    from numpy import ndarray

    def __init__(
        self,
        max_num_of_steps: int,
        width: int,
        height: int,
        num_of_swarm_agents: int,
        random_agent_per_step: bool,
        max_new_opinion_weighting: float,
        num_of_malicious_agents: int,
        sensing_noise: float,
        communication_noise: float,
        **kwargs,
    ):

        super(DynamicOpinionWeightingTrainer, self).__init__()

        self.max_num_steps = max_num_of_steps
        self.width, self.height = width, height
        self.num_of_swarm_agents = num_of_swarm_agents
        self.random_agent_per_step = random_agent_per_step
        self.max_new_opinion_weighting = max_new_opinion_weighting
        self.num_of_malicious_agents = num_of_malicious_agents
        self.sensing_noise = sensing_noise
        self.communication_noise = communication_noise

        self.environment_type_weighting = [33, 33, 33]
        self.model = None

        from numpy import float32
        from gym import spaces

        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=float32)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(2,), dtype=float32
        )

    def set_model(self, model: Union[PPO, DQN]):
        self.model = model

    def calculate_reward(self, agent: SwarmAgent) -> int:
        self.correct_opinion_weighting = agent.opinion_weights[self.correct_opinion]
        self.incorrect_opinion_weighting = agent.opinion_weights[
            (self.correct_opinion + 1) % 2
        ]
        return (
            1
            / ((self.max_new_opinion_weighting + 0.01) - self.correct_opinion_weighting)
        ) + (1 / (0.01 + self.incorrect_opinion_weighting))

    def transform_action_to_opinion_weighting(self, action: ndarray) -> ndarray:
        return (self.max_new_opinion_weighting / 2) * (action + 1)

    def return_action_for_other_agent(self, agent: SwarmAgent):
        if self.model is not None:
            return self.transform_action_to_opinion_weighting(
                self.model.predict(agent.return_opinion_weight_states())[0]
            )
        return [self.max_new_opinion_weighting, self.max_new_opinion_weighting]

    def step(self, action):
        for agent in self.malicious_agents:
            agent.navigate(self.tile_grid)

        for agent in self.swarm_agents:
            if agent != self.agent_to_train:
                agent.opinion_weights = self.return_action_for_other_agent(agent)
            else:
                agent.opinion_weights = self.transform_action_to_opinion_weighting(
                    action
                )

                reward = self.calculate_reward(agent)

            agent.decide_to_sense_or_broadcast()
            agent.navigate_and_recieve_opinions(self.tile_grid)

        wandb.log(
            {
                "correct_opinion_weighting": self.correct_opinion_weighting,
                "incorrect_opinion_weighting": self.incorrect_opinion_weighting,
                "distance_from_optimal_weighting": (
                    (self.max_new_opinion_weighting - self.correct_opinion_weighting)
                    + self.incorrect_opinion_weighting
                ),
            }
        )

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True

            self.environment_type_weighting[self.index_of_environment] = round(
                (
                    inverse_sigmoid_for_weighting(
                        (
                            self.correct_opinion_weighting
                            / self.max_new_opinion_weighting
                        )
                        * 100
                    )
                    + sigmoid_for_weighting(
                        (
                            self.incorrect_opinion_weighting
                            / self.max_new_opinion_weighting
                        )
                        * 100
                    )
                )
                / 2.0
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

        (
            self.index_of_environment,
            self.tile_grid,
        ) = return_environment_based_on_weighting_list(
            self.environment_type_weighting, self.width, self.height
        )

        self.correct_opinion = round(
            return_ratio_of_white_to_black_tiles(self.tile_grid)
        )

        list_of_coordinates_to_distribute_agents_over = (
            return_list_of_coordinates_column_by_columns(
                num_of_columns=self.width, num_of_rows=self.height
            )
        )

        self.swarm_agents = [
            SwarmAgent(
                starting_cell=(
                    self.tile_grid[list_of_coordinates_to_distribute_agents_over.pop(0)]
                ),
                needs_models_loaded=True,
                current_direction_facing=random.randint(0, 3),
                max_new_opinion_weighting=self.max_new_opinion_weighting,
                sensing_noise=self.sensing_noise,
                communication_noise=self.communication_noise,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.malicious_agents = [
            MaliciousAgent(
                starting_cell=(
                    self.tile_grid[list_of_coordinates_to_distribute_agents_over.pop(0)]
                ),
                malicious_opinion=((self.correct_opinion + 1) % 2),
                current_direction_facing=random.randint(0, 3),
            )
            for _ in range(self.num_of_malicious_agents)
        ]

        self.agent_to_train = random.choice(self.swarm_agents)

        return self.agent_to_train.return_opinion_weight_states()
