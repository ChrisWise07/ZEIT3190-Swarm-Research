import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from .tile_properties import TileColour, WallType


@dataclass(repr=False, eq=False)
class SwarmAgent:
    id: int
    current_cell: Tuple[int, int] = (-1, -1)
    current_opinion: float = field(init=False)
    calculated_collective_opinion: float = field(init=False)
    cells_visited: Set[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        self.cells_visited = set()

    def occupy_cell(self, tile: Dict):
        if not (tile["occupied"]):
            self.current_cell = tile["id"]
            tile["occupied"] = True

    def leave_cell(self, tile: Dict):
        tile["occupied"] = False
        self.cells_visited.add(tile["id"])

    def return_reward(self):
        return int(not (self.current_cell in self.cells_visited))
