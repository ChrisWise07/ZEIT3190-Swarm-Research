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
    needs_models_loaded: InitVar[bool] = False
    current_direction_facing: int = Direction.RIGHT.value
    navigation_model = None
    sense_broadcast_model = None
    sensing: bool = True
    num_of_white_cells_observed: int = 0
    num_of_cells_observed: int = 0
    collective_opinion_weight: float = 0.9
    calculated_collective_opinion: float = 0.5
    communication_range: int = 1
    current_cell: Tuple[int, int] = field(init=False)
    cells_visited: Set[Tuple[int, int]] = field(init=False)

    def __post_init__(
        self, starting_cell: Dict[str, Any], needs_models_loaded: bool
    ) -> None:
        self.cells_visited = set()
        if needs_models_loaded:
            from stable_baselines3 import PPO

            self.navigation_model = PPO.load(
                f"{TRAINED_MODELS_DIRECTORY}/multi_agent_nav_model"
            )

            self.sense_broadcast_model = PPO.load(
                f"{TRAINED_MODELS_DIRECTORY}/sense_broadcast_model"
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
                if tile["colour"]:
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
            return -1 / len(self.cells_visited)
        return len(self.cells_visited)

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

    def calculate_opinion(self) -> int:
        return round(self.num_of_white_cells_observed / self.num_of_cells_observed)

    def return_opinion(self) -> Union[int, None]:
        if not (self.sensing):
            return self.calculate_opinion()

    def update_calculated_collective_opinion(self, opinion: int) -> None:
        self.calculated_collective_opinion = (
            self.collective_opinion_weight * self.calculated_collective_opinion
            + (1 - self.collective_opinion_weight) * opinion
        )

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
            if tile["agent"] and tile["agent"] != self:
                recieved_opinion = tile["agent"].return_opinion()
                if recieved_opinion is not None:
                    self.update_calculated_collective_opinion(recieved_opinion)

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
        self.sensing = bool(self.choose_sense_broadcast_action())

    def perform_decision_navigate_opinion_update_cycle(
        self, tile_grid: np.ndarray
    ) -> None:
        self.decide_to_sense_or_broadcast()
        self.navigate(tile_grid=tile_grid)
        self.recieve_local_opinions(tile_grid=tile_grid)
        # decide if done add here
