from dataclasses import dataclass, field
from typing import List
from .tile import Tile
from .tile_properties import TileColour, WallType
import numpy as np


@dataclass(repr=False, eq=False)
class TiledEnvironmentClass:
    height: int
    width: int
    ratio_of_white_to_black_tiles: float
    clustered: True
    tile_grid: np.ndarray = field(init=False)

    def __post_init__(self):
        self.tile_grid = np.empty((self.height, self.width), dtype=object)
        for row in range(self.height):
            for column in range(self.width):
                if row == 0 and column == 0:
                    self.tile_grid[row, column] = Tile(
                        colour=TileColour.BLACK,
                        walls=[WallType.LEFT_WALL, WallType.TOP_WALL],
                    )
                if row == 0 and column == (self.width - 1):
                    self.tile_grid[row, column] = Tile(
                        colour=TileColour.BLACK,
                        walls=[WallType.RIGHT_WALL, WallType.TOP_WALL],
                    )

        # test_tile = Tile(colour=TileColour.BLACK, tile_type=TileType.LEFT_CORNER)
        # self.tile_grid = [[test_tile]]
