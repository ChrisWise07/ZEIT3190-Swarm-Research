import re
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from .tile_properties import TileColour, WallType
from .swarm_agent_enums import Direction


@dataclass(repr=False, eq=False)
class SwarmAgent:
    id: int
    current_cell: Tuple[int, int] = (-1, -1)
    current_direction_facing: Direction = Direction.RIGHT
    current_opinion: float = field(init=False)
    calculated_collective_opinion: float = field(init=False)
    cells_visited: Set[Tuple[int, int]] = field(init=False)

    def __post_init__(self) -> None:
        self.cells_visited = set()

    def occupy_cell(self, tile: Dict) -> bool:
        if not (tile["occupied"]):
            self.current_cell = tile["id"]
            tile["occupied"] = True
            return True
        else:
            return False

    def leave_cell(self, tile: Dict) -> None:
        tile["occupied"] = False
        self.cells_visited.add(tile["id"])

    def return_reward(self) -> int:
        return int(not (self.current_cell in self.cells_visited))

    def __return_new_cell_coordinate(self) -> Tuple[int, int]:
        return {
            Direction.RIGHT: (self.current_cell[0], self.current_cell[1] + 1),
            Direction.DOWN: (self.current_cell[0] + 1, self.current_cell[1]),
            Direction.LEFT: (self.current_cell[0], self.current_cell[1] - 1),
            Direction.UP: (self.current_cell[0] - 1, self.current_cell[1]),
        }[self.current_direction_facing]

    def forward_step(self, tile_grid: np.ndarray) -> None:

        new_cell = self.__return_new_cell_coordinate()
        old_cell = self.current_cell

        if self.occupy_cell(tile=tile_grid[new_cell]):
            self.leave_cell(tile=tile_grid[old_cell])
        else:
            self.cells_visited.add(tile_grid[old_cell]["id"])
