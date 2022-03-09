import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .tile_properties import TileColour, WallType


@dataclass(repr=False, eq=False)
class Agent:
    id: int
    current_opinion: float = field(init=False)
    calculated_collective_opinion: float = field(init=False)
    cells_visited: List[Tuple[int, int]] = field(init=False)

    def __post_init__(self):
        pass