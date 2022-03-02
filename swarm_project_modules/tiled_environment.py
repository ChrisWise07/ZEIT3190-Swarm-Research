import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .tile import Tile
from .tile_properties import TileColour, WallType
from .regex_dictionary import RegexDict


@dataclass(repr=False, eq=False)
class TiledEnvironmentClass:
    height: int
    width: int
    ratio_of_white_to_black_tiles: float
    clustered: True
    tile_grid: np.ndarray = field(init=False)
    tile_walls_to_coordinates_map: Dict[str, List[WallType]] = field(init=False)

    def return_tile_with_walls_based_on_coordinates(
        self, coordinates: Tuple[int, int], colour: TileColour
    ) -> Tile:
        return Tile(
            colour=colour,
            walls=self.tile_walls_to_coordinates_map.get(str(coordinates)),
        )

    def return_coordinate_to_walls_dict(self):
        return RegexDict(
            {
                # top left corner
                "(0, 0)": [WallType.LEFT_WALL, WallType.TOP_WALL],
                # top right corner
                f"(0, {(self.width - 1)})": [WallType.RIGHT_WALL, WallType.TOP_WALL],
                # bottom left corner
                f"({(self.height - 1)}, 0)": [WallType.LEFT_WALL, WallType.BOTTOM_WALL],
                # bottom right corner
                f"({(self.height - 1)}, {(self.width - 1)})": [
                    WallType.RIGHT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                # top_edge
                "(0, .)": [WallType.TOP_WALL],
                # left_wall
                "(., 0)": [WallType.LEFT_WALL],
                # right_wall
                f"(., {(self.width - 1)})": [WallType.RIGHT_WALL],
                # bottom_edge
                f"({(self.height - 1)}, .)": [WallType.BOTTOM_WALL],
                # all other tiles
                "(., .)": [],
            }
        )

    def __post_init__(self):
        self.tile_walls_to_coordinates_map = self.return_coordinate_to_walls_dict()

        self.tile_grid = np.empty((self.height, self.width), dtype=object)

        for row in range(self.height):
            for column in range(self.width):
                self.tile_grid[
                    row, column
                ] = self.return_tile_with_walls_based_on_coordinates(
                    coordinates=(row, column), colour=TileColour.WHITE
                )
