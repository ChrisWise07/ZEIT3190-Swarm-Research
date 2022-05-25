import os
import sys
from typing import List
import gym
from numpy import average
import wandb
import random

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
            low=0.0, high=float(max_num_of_steps), shape=(4,), dtype=float32
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
            return -1

        if agent.calculate_opinion() != self.correct_opinion:
            return -200

        return 100

    def return_action_for_other_agent(self, agent: SwarmAgent):
        if self.model is not None:
            return self.model.predict(agent.return_commit_decision_states())[0].item()
        return random.choice([0, 1])

    def step(self, action):

        for agent in self.uncommited_agents[:]:
            if agent != self.agent_to_train:
                agent.committed_to_opinion = self.return_action_for_other_agent(agent)
            else:
                agent.committed_to_opinion = action
                self.reward = self.calculate_reward(agent)

            if agent.committed_to_opinion:
                self.commited_agents.append(agent)
                self.uncommited_agents.remove(agent)

        for agent in self.swarm_agents:
            agent.perform_decision_navigate_opinion_update_cycle(self.tile_grid)

        self.num_steps += 1

        if self.num_steps == self.max_num_steps:
            self.done = True

            correct_commitment_count = 0
            total_commitment_time = 0

            for agent in self.commited_agents:
                if agent.calculate_opinion() == self.correct_opinion:
                    correct_commitment_count += 1
                total_commitment_time += agent.num_of_cycles_performed

            correct_commitment_ratio = (
                correct_commitment_count / self.num_of_swarm_agents
            )

            wandb.log(
                {
                    "percentage_of_correct_commitments": correct_commitment_ratio,
                    "average_commitment_time": total_commitment_time
                    / self.num_of_swarm_agents,
                }
            )

            self.environment_type_weighting[self.index_of_environment] = round(
                (inverse_sigmoid_for_weighting(correct_commitment_ratio * 100))
            )

        if self.random_agent_per_step:
            if len(self.uncommited_agents):  # if there are agents left to commit
                self.agent_to_train = random.choice(self.uncommited_agents)

        return (
            self.agent_to_train.return_commit_decision_states(),
            self.reward,
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
                current_direction_facing=random.randint(0, 3),
                needs_models_loaded=True,
            )
            for _ in range(self.num_of_swarm_agents)
        ]

        self.uncommited_agents = self.swarm_agents.copy()
        self.commited_agents = []  # type: List[SwarmAgent]

        self.agent_to_train = random.choice(self.uncommited_agents)

        return self.agent_to_train.return_commit_decision_states()
