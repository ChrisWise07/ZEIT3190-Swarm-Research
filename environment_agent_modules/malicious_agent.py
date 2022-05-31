import random
import numpy as np
from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple
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
class MaliciousAgent:
    starting_cell: InitVar[Dict[str, Any]]
    malicious_opinion: int
    current_direction_facing: int = Direction.RIGHT.value
    current_cell: Tuple[int, int] = field(init=False)

    def __post_init__(
        self,
        starting_cell: Dict[str, Any],
    ) -> None:
        self.cells_visited = set()

        from stable_baselines3 import PPO

        self.navigation_model = PPO.load(
            f"{TRAINED_MODELS_DIRECTORY}/{'multi_agent_nav_model'}"
        )

        if not (self.occupy_cell(starting_cell)):
            self.current_cell = (None, None)

    def occupy_cell(self, tile: Dict[str, Any]) -> bool:
        if not (tile["agent"]):
            tile["agent"] = self
            self.current_cell = tile["id"]
            return True

        return False

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

    def return_opinion(self) -> int:
        return self.malicious_opinion
