from numpy import ndarray

from dataclasses import dataclass, field, InitVar
from typing import Any, Dict, List, Tuple, Set
from .tile_properties import TileColour, WallType
from .swarm_agent_enums import Direction, Turn, ObjectType, Position


@dataclass(repr=False, eq=False)
class SwarmAgent:
    id: int
    starting_cell: InitVar[Dict[str, Any]]
    current_direction_facing: Direction = Direction.RIGHT
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

    def return_reward(self) -> int:
        return int(not (self.current_cell in self.cells_visited))

    def __return_new_cell_coordinate(self) -> Tuple[int, int]:
        return {
            Direction.RIGHT: (self.current_cell[0], self.current_cell[1] + 1),
            Direction.DOWN: (self.current_cell[0] + 1, self.current_cell[1]),
            Direction.LEFT: (self.current_cell[0], self.current_cell[1] - 1),
            Direction.UP: (self.current_cell[0] - 1, self.current_cell[1]),
        }[self.current_direction_facing]

    def validate_cell(
        self, new_cell: Tuple[int, int], grid_shape: Tuple[int, int]
    ) -> bool:
        return all(
            [
                (row_or_col >= 0 and row_or_col < dimension)
                for row_or_col, dimension in zip(new_cell, grid_shape)
            ]
        )

    def forward_step(self, tile_grid: ndarray) -> None:
        new_cell = self.__return_new_cell_coordinate()
        old_cell = self.current_cell

        if self.validate_cell(
            new_cell=new_cell, grid_shape=tile_grid.shape
        ) and self.occupy_cell(tile=tile_grid[new_cell]):
            self.leave_cell(tile=tile_grid[old_cell])
        else:
            self.add_cell_to_visited_list(old_cell)

    def turn(self, turn_type: Turn) -> None:
        self.current_direction_facing = Direction(
            (self.current_direction_facing.value + turn_type.value) % 4
        )

    def __return_object_type_based_on_num_wall(self, num_of_walls: int) -> ObjectType:
        return {0: ObjectType.NONE, 1: ObjectType.WALL, 2: ObjectType.CORNER}[
            num_of_walls
        ]

    def get_type_of_nearest_object(self, tile_grid: ndarray) -> ObjectType:
        tile_walls = tile_grid[self.current_cell]["walls"]

        if len(tile_walls):
            return self.__return_object_type_based_on_num_wall(
                num_of_walls=len(tile_walls)
            )
        else:
            # agents have precedence over walls when detecting
            # objects on the next tile along
            next_tile_along = tile_grid[self.__return_new_cell_coordinate()]

            if next_tile_along["occupied"]:
                return ObjectType.AGENT
            else:
                next_tile_along_walls = tile_grid[self.__return_new_cell_coordinate()][
                    "walls"
                ]
                return self.__return_object_type_based_on_num_wall(
                    num_of_walls=len(next_tile_along_walls)
                )

    def get_relative_position_of_object(self) -> Position:
        return Position.LEFT
