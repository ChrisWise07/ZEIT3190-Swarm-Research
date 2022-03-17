import numpy as np

from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple, Set
from .tile_properties import WallType
from .utils import validate_cell
from .swarm_agent_enums import (
    Direction,
    RelativeMotion,
    Turn,
    ObjectType,
    RelativePosition,
)


@dataclass(repr=False, eq=False)
class SwarmAgent:
    id: int
    starting_cell: InitVar[Dict[str, Any]]
    current_direction_facing: int = Direction.RIGHT.value
    current_cell: Tuple[int, int] = field(init=False)
    current_opinion: float = field(init=False)
    calculated_collective_opinion: float = field(init=False)
    cells_visited: Set[Tuple[int, int]] = field(init=False)

    def __post_init__(self, starting_cell: Dict[str, Any]) -> None:
        self.cells_visited = set()
        if not (self.occupy_cell(starting_cell)):
            self.current_cell = (None, None)

    def add_cell_to_visited_list(self, cell_id: Tuple[int, int]) -> None:
        self.cells_visited.add(cell_id)

    def occupy_cell(self, tile: Dict[str, Any]) -> bool:
        if not (tile["occupied"]):
            tile["occupied"] = True
            self.current_cell = tile["id"]
            return True
        else:
            return False

    def leave_cell(self, tile: Dict) -> None:
        tile["occupied"] = False
        self.add_cell_to_visited_list(tile["id"])

    def return_navigation_reward(self) -> int:
        return int(not (self.current_cell in self.cells_visited))

    def __return_next_cell_coordinate(self) -> Tuple[int, int]:
        return {
            Direction.RIGHT.value: (self.current_cell[0], self.current_cell[1] + 1),
            Direction.DOWN.value: (self.current_cell[0] + 1, self.current_cell[1]),
            Direction.LEFT.value: (self.current_cell[0], self.current_cell[1] - 1),
            Direction.UP.value: (self.current_cell[0] - 1, self.current_cell[1]),
        }[self.current_direction_facing]

    def forward_step(self, tile_grid: np.ndarray) -> bool:
        new_cell = self.__return_next_cell_coordinate()
        old_cell = self.current_cell

        if validate_cell(
            new_cell=new_cell, grid_shape=tile_grid.shape
        ) and self.occupy_cell(tile=tile_grid[new_cell]):
            tile_grid[old_cell]["occupied"] = False

    def turn(self, turn_type: int) -> None:
        self.current_direction_facing = (self.current_direction_facing + turn_type) % 4

    def __get_relative_position_of_object(self, tile_walls: List[WallType]) -> int:
        if len(tile_walls) >= 2:
            tile_walls = [
                wall
                for wall in tile_walls
                if ((self.current_direction_facing - wall) % 4 in [1, 3])
            ]

        return (self.current_direction_facing - tile_walls[0]) % 4

    def __get_relative_motion_of_object(self, tile_walls: List[WallType]) -> int:
        if self.current_direction_facing in tile_walls:
            return RelativeMotion.APPROACHING.value

        return RelativeMotion.ESCAPING.value

    def __call_each_state_function_for_tile(
        self, tile_walls: List[WallType]
    ) -> Tuple[int, int, int]:
        if not (len(tile_walls)):
            return (
                ObjectType.NONE.value,
                RelativeMotion.APPROACHING.value,
                RelativePosition.FRONT.value,
            )
        else:
            return (
                len(tile_walls),
                self.__get_relative_motion_of_object(tile_walls=tile_walls),
                self.__get_relative_position_of_object(tile_walls=tile_walls),
            )

    def get_navigation_states(self, tile_grid: np.ndarray) -> Tuple[int, int, int]:

        current_tile = tile_grid[self.current_cell]
        next_tile_coordinates = self.__return_next_cell_coordinate()

        if validate_cell(next_tile_coordinates, grid_shape=tile_grid.shape):
            next_tile_along = tile_grid[next_tile_coordinates]
            # agents have precedence over walls when detecting
            # objects on the next tile along
            if next_tile_along["occupied"]:
                return (
                    ObjectType.AGENT.value,
                    RelativeMotion.APPROACHING.value,
                    RelativePosition.FRONT.value,
                )

        # only corners and edges will not have a next cell along if facing into them
        # and the statement below will be true then, so no need for next_tile_along
        if current_tile["walls"]:
            return self.__call_each_state_function_for_tile(
                tile_walls=current_tile["walls"]
            )
        else:
            return self.__call_each_state_function_for_tile(
                tile_walls=next_tile_along["walls"]
            )

    def perform_navigation_action(self, action: int, tile_grid: np.ndarray) -> None:
        self.add_cell_to_visited_list(self.current_cell)

        if action == 0:
            self.forward_step(tile_grid=tile_grid)
        elif action == 1:
            self.turn(turn_type=Turn.RIGHT.value)
        elif action == 2:
            self.turn(turn_type=Turn.LEFT.value)
