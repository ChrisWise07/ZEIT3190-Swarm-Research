import random
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple, Set, Union
from .tile_properties import WallType
from .utils import validate_cell
from helper_files import TRAINED_MODELS_DIRECTORY
from .swarm_agent_enums import (
    Direction,
    Turn,
    ObjectType,
    RelativePosition,
)


@dataclass(repr=False, eq=False)
class SwarmAgent:
    starting_cell: InitVar[Dict[str, Any]]
    needs_models_loaded: InitVar[bool]
    opinion_weighting_method: InitVar[str] = "list_of_weights"
    model_names: InitVar[Dict[str, Union[str, None]]] = {
        "nav_model": "multi_agent_nav_model",
        "sense_model": "sense_broadcast_model",
        "commit_to_opinion_model": None,
        "dynamic_opinion_model": None,
    }

    current_direction_facing: int = Direction.RIGHT.value
    max_new_opinion_weighting: float = 0.1
    sensing_noise: float = 0.0
    communication_noise: float = 0.0

    communication_range: int = 1
    sensing: int = 1
    committed_to_opinion: int = 0
    num_of_white_cells_observed: int = 0
    num_of_cells_observed: int = 0
    calculated_collective_opinion: float = 0.5

    current_cell: Tuple[int, int] = field(init=False)
    cells_visited: Set[Tuple[int, int]] = field(init=False)

    def __post_init__(
        self,
        starting_cell: Dict[str, Any],
        needs_models_loaded: bool,
        opinion_weighting_method: str,
        model_names: Dict[str, Union[str, None]],
    ) -> None:
        self.cells_visited = set()

        self.opinion_weights = [
            self.max_new_opinion_weighting,
            self.max_new_opinion_weighting,
        ]

        self.opinion_weight_function = {
            "list_of_weights": self.return_opinion_weight_based_on_opinion_weight_list,
            "equation_based": self.return_opinion_weight_based_on_equation,
        }[opinion_weighting_method]

        if needs_models_loaded:
            from stable_baselines3 import PPO, DQN

            nav_model = model_names.get("nav_model")
            if nav_model is not None:
                self.navigation_model = PPO.load(
                    f"{TRAINED_MODELS_DIRECTORY}/{nav_model}"
                )

            sense_model = model_names.get("sense_model")
            if sense_model is not None:
                self.sense_broadcast_model = DQN.load(
                    f"{TRAINED_MODELS_DIRECTORY}/{model_names.get('sense_model')}"
                )

            commit_to_opinion_model = model_names.get("commit_to_opinion_model")
            if commit_to_opinion_model is not None:
                self.commit_to_opinion_model = DQN.load(
                    f"{TRAINED_MODELS_DIRECTORY}/{model_names.get('commit_to_opinion_model')}"
                )

            dynamic_opinion_model = model_names.get("dynamic_opinion_model")
            if dynamic_opinion_model is not None:
                self.dynamic_opinion_model = PPO.load(
                    f"{TRAINED_MODELS_DIRECTORY}/{model_names.get('dynamic_opinion_model')}"
                )

        if not (self.occupy_cell(starting_cell)):
            self.current_cell = (None, None)

    def add_cell_to_visited_list(self, cell_id: Tuple[int, int]) -> None:
        self.cells_visited.add(cell_id)

    def occupy_cell(self, tile: Dict[str, Any]) -> bool:
        if not (tile["agent"]):
            tile["agent"] = self
            self.current_cell = tile["id"]
            if self.sensing:
                self.num_of_cells_observed += 1
                tile_color = tile["colour"]
                tile_color = random.choices(
                    [tile_color, (tile_color + 1) % 2],
                    [1 - self.sensing_noise, self.sensing_noise],
                    k=1,
                )[0]
                if tile_color:
                    self.num_of_white_cells_observed += 1
            return True

        return False

    def return_num_of_cells_visited(self) -> int:
        if self.current_cell in self.cells_visited:
            return len(self.cells_visited)
        return len(self.cells_visited) + 1

    def leave_cell(self, tile: Dict) -> None:
        tile["agent"] = None
        self.add_cell_to_visited_list(tile["id"])

    def return_navigation_reward(self) -> int:
        if self.current_cell in self.cells_visited:
            return -1 / self.return_num_of_cells_visited()
        return 1 * self.return_num_of_cells_visited()

    def __return_next_cell_coordinate(self) -> Tuple[int, int]:
        return {
            Direction.RIGHT.value: (self.current_cell[0], self.current_cell[1] + 1),
            Direction.DOWN.value: (self.current_cell[0] + 1, self.current_cell[1]),
            Direction.LEFT.value: (self.current_cell[0], self.current_cell[1] - 1),
            Direction.UP.value: (self.current_cell[0] - 1, self.current_cell[1]),
        }[self.current_direction_facing]

    def forward_step(self, tile_grid: np.ndarray) -> None:
        new_cell = self.__return_next_cell_coordinate()
        old_cell = self.current_cell

        if validate_cell(
            new_cell=new_cell, grid_shape=tile_grid.shape
        ) and self.occupy_cell(tile=tile_grid[new_cell]):
            tile_grid[old_cell]["agent"] = None

    def turn(self, turn_type: int) -> None:
        self.current_direction_facing = (self.current_direction_facing + turn_type) % 4

    def __call_each_state_function_for_tile(
        self, tile_walls: List[WallType]
    ) -> Tuple[int, int]:
        return {
            0: lambda _: (ObjectType.NONE.value, RelativePosition.FRONT.value),
            1: lambda tile_walls: (
                ObjectType.WALL.value,
                (self.current_direction_facing - tile_walls[0]) % 4,
            ),
            2: lambda tile_walls: (
                ObjectType.CORNER.value,
                (
                    self.current_direction_facing
                    - [
                        wall
                        for wall in tile_walls
                        if ((self.current_direction_facing - wall) % 4 in [1, 3])
                    ][0]
                )
                % 4
                + 1,
            ),
        }[len(tile_walls)](tile_walls)

    def get_navigation_states(self, tile_grid: np.ndarray) -> Tuple[int, int]:
        next_tile_coordinates = self.__return_next_cell_coordinate()

        if validate_cell(new_cell=next_tile_coordinates, grid_shape=tile_grid.shape):
            next_tile_along = tile_grid[next_tile_coordinates]
            # other agents have precedence when detecting objects on the next tile along
            if next_tile_along["agent"]:
                return (
                    ObjectType.AGENT.value,
                    RelativePosition.FRONT.value,
                )

            return self.__call_each_state_function_for_tile(
                tile_walls=next_tile_along["walls"]
            )

        # agent is facing into a corner or wall
        return self.__call_each_state_function_for_tile(
            tile_walls=tile_grid[self.current_cell]["walls"]
        )

    def perform_navigation_action(self, action: int, tile_grid: np.ndarray) -> None:
        self.add_cell_to_visited_list(self.current_cell)

        if isinstance(action, (np.ndarray)):
            action = action[0]

        return {
            0: lambda self, tile_grid: (self.forward_step(tile_grid=tile_grid)),
            1: lambda self, _: self.turn(turn_type=Turn.LEFT.value),
            2: lambda self, _: self.turn(turn_type=Turn.RIGHT.value),
        }[action](self, tile_grid)

    def choose_navigation_action(self, tile_grid: np.ndarray) -> int:
        return self.navigation_model.predict(
            np.array([self.get_navigation_states(tile_grid=tile_grid)])
        )[0].item()

    def navigate(self, tile_grid: np.ndarray) -> None:
        self.perform_navigation_action(
            action=self.choose_navigation_action(tile_grid=tile_grid),
            tile_grid=tile_grid,
        )

    def new_choose_navigation_action(self, tile_grid: np.ndarray) -> int:
        next_tile_coordinates = self.__return_next_cell_coordinate()

        if validate_cell(new_cell=next_tile_coordinates, grid_shape=tile_grid.shape):
            next_tile_along = tile_grid[next_tile_coordinates]

            if next_tile_along["agent"]:
                return random.choice([1, 2])

            return 0

        # agent is facing into a corner or wall
        tile_walls = tile_grid[self.current_cell]["walls"]

        if len(tile_walls) == 1:
            return random.choice([1, 2])

        return 1 if tile_walls[0] == self.current_direction_facing else 2

    def new_navigate(self, tile_grid: np.ndarray) -> None:
        self.perform_navigation_action(
            action=self.new_choose_navigation_action(tile_grid=tile_grid),
            tile_grid=tile_grid,
        )

    def calculate_opinion(self) -> int:
        return round(self.num_of_white_cells_observed / self.num_of_cells_observed)

    def return_opinion(self) -> Union[int, None]:
        if not (self.sensing):
            return self.calculate_opinion()

    def return_opinion_weight_based_on_opinion_weight_list(self, opinion: int) -> float:
        return self.opinion_weights[opinion]

    def return_opinion_weight_based_on_equation(self, opinion: int) -> float:
        return self.max_new_opinion_weighting * (
            1 - abs(self.calculated_collective_opinion - opinion)
        )

    def update_collective_opinion(self, opinion: int) -> None:
        opinion_weight = self.opinion_weight_function(opinion=opinion)

        opinion = random.choices(
            [opinion, (opinion + 1) % 2],
            [1 - self.communication_noise, self.communication_noise],
            k=1,
        )[0]

        self.calculated_collective_opinion = (
            (1 - opinion_weight) * self.calculated_collective_opinion
        ) + (opinion_weight * opinion)

    def recieve_local_opinions(self, tile_grid: np.ndarray):
        current_y, current_x = self.current_cell

        communication_y_min = max(0, current_y - self.communication_range)
        communication_x_min = max(0, current_x - self.communication_range)
        communication_y_max = min(
            tile_grid.shape[0], current_y + self.communication_range
        )
        communication_x_max = min(
            tile_grid.shape[1], current_y + self.communication_range
        )

        local_area = tile_grid[
            communication_y_min : communication_y_max + 1,
            communication_x_min : communication_x_max + 1,
        ]

        for tile in local_area.flat:
            if tile["agent"]:
                recieved_opinion = tile["agent"].return_opinion()
                if recieved_opinion is not None:
                    self.update_collective_opinion(recieved_opinion)

    def return_sense_broadcast_states(self) -> np.ndarray:
        opinion = self.calculate_opinion()
        return np.array(
            (
                self.num_of_cells_observed,
                opinion,
                self.calculated_collective_opinion,
                abs(self.calculated_collective_opinion - opinion),
            ),
            dtype=np.float32,
        )

    def choose_sense_broadcast_action(self) -> int:
        return self.sense_broadcast_model.predict(self.return_sense_broadcast_states())[
            0
        ].item()

    def decide_to_sense_or_broadcast(self) -> None:
        self.sensing = self.choose_sense_broadcast_action()

    def return_commit_decision_states(self) -> np.ndarray:
        return np.array(
            (
                self.num_of_cells_observed,
                abs(self.calculated_collective_opinion - self.calculate_opinion()),
            ),
            dtype=np.float32,
        )

    def return_opinion_weight_states(self) -> np.ndarray:
        return np.array(
            (
                self.calculate_opinion(),
                self.calculated_collective_opinion,
            ),
            dtype=np.float32,
        )

    def choose_commit_decision_action(self) -> int:
        return self.commit_to_opinion_model.predict(
            self.return_commit_decision_states()
        )[0].item()

    def decide_if_to_commit(self) -> None:
        if not self.committed_to_opinion:
            self.committed_to_opinion = self.choose_commit_decision_action()

    def navigate_and_recieve_opinions(self, tile_grid: np.ndarray) -> None:
        self.navigate(tile_grid=tile_grid)
        self.recieve_local_opinions(tile_grid=tile_grid)

    def predict_optimal_opinion_weights(self) -> None:
        return self.dynamic_opinion_model.predict(self.return_opinion_weight_states())[
            0
        ]

    def transform_action_to_opinion_weighting(self, action: np.ndarray) -> np.ndarray:
        return (self.max_new_opinion_weighting / 2) * (action + 1)

    def set_dynamic_opinion_weights(self) -> None:
        self.opinion_weights = self.transform_action_to_opinion_weighting(
            self.predict_optimal_opinion_weights()
        )

    def perform_decision_navigate_opinion_update_cycle(
        self, tile_grid: np.ndarray
    ) -> None:

        if not (self.committed_to_opinion):
            self.decide_to_sense_or_broadcast()
            self.navigate_and_recieve_opinions(tile_grid=tile_grid)
        else:
            self.sensing = 0  # agent is broadcasting opinion
            self.navigate(tile_grid=tile_grid)
