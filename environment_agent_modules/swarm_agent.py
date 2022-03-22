import numpy as np

from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple, Set
from .tile_properties import WallType
from .utils import validate_cell
from .swarm_agent_enums import (
    Direction,
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

    def forward_step(self, tile_grid: np.ndarray) -> None:
        new_cell = self.__return_next_cell_coordinate()
        old_cell = self.current_cell

        if validate_cell(
            new_cell=new_cell, grid_shape=tile_grid.shape
        ) and self.occupy_cell(tile=tile_grid[new_cell]):
            tile_grid[old_cell]["occupied"] = False

    def turn(self, turn_type: int) -> None:
        self.current_direction_facing = (self.current_direction_facing + turn_type) % 4

    def __call_each_state_function_for_tile(
        self, tile_walls: List[WallType]
    ) -> Tuple[int, int, int]:
        return {
            0: lambda _: (ObjectType.NONE.value, RelativePosition.FRONT.value),
            1: lambda tile_walls: (
                ObjectType.WALL.value,
                (self.current_direction_facing - tile_walls[0]) % 4,
            ),
            2: lambda _: (
                ObjectType.CORNER.value,
                RelativePosition.FRONT.value,
            ),
        }[len(tile_walls)](tile_walls)

    def get_navigation_states(self, tile_grid: np.ndarray) -> Tuple[int, int]:
        next_tile_coordinates = self.__return_next_cell_coordinate()

        if validate_cell(new_cell=next_tile_coordinates, grid_shape=tile_grid.shape):
            next_tile_along = tile_grid[next_tile_coordinates]
            # other agents have precedence when detecting objects on the next tile along
            if next_tile_along["occupied"]:
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

        return {
            0: lambda self, tile_grid: (self.forward_step(tile_grid=tile_grid)),
            1: lambda self, _: self.turn(turn_type=Turn.LEFT.value),
            2: lambda self, _: self.turn(turn_type=Turn.RIGHT.value),
        }[action](self, tile_grid)
