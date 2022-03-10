import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set
from .tile_properties import TileColour, WallType


@dataclass(repr=False, eq=False)
class SwarmAgent:
    id: int
    current_opinion: float = field(init=False)
    calculated_collective_opinion: float = field(init=False)
    cells_visited: Set[Tuple[int,int]] = field(init=False)
    current_cell: Tuple[int, int] = field(init=False)

    def __post_init__(self):
        self.cells_visited = set()

    def occupy_cell(self, tile: Dict):
        if not(tile["occupied"]):
            self.cells_visited.add(tile["id"])
            self.current_cell = tile["id"]
            tile["occupied"] = True

    def leave_cell(self, tile: Dict):
        tile["occupied"] = False
    