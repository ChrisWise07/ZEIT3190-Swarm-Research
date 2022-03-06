import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from .tile import Tile
from .tile_properties import TileColour, WallType
from .regex_dictionary import RegexDict


@dataclass(repr=False, eq=False)
class TiledEnvironment:
    height: int
    width: int
    ratio_of_white_to_black_tiles: float
    clustered: bool = False
    initial_observations_helpful: bool = True
    tile_grid: np.ndarray = field(init=False)

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

    def non_clustered_tile_grid(
        self, tile_walls_to_coordinates_map: Dict[str, List[WallType]]
    ):
        random_numbers_between_0_1 = np.random.rand(self.height, self.width)
        for row in range(self.height):
            for column in range(self.width):
                if (
                    random_numbers_between_0_1[(row, column)]
                    < self.ratio_of_white_to_black_tiles
                ):
                    colour = TileColour.WHITE
                else:
                    colour = TileColour.BLACK

                self.tile_grid[row, column] = {
                    "colour": colour,
                    "walls": tile_walls_to_coordinates_map.get(str((row, column))),
                    "occupied": False,
                }

    def clustered_environment(
        self, tile_walls_to_coordinates_map: Dict[str, List[WallType]]
    ):
        if self.initial_observations_helpful:
            pass

    def __post_init__(self):
        tile_walls_to_coordinates_map = self.return_coordinate_to_walls_dict()

        self.tile_grid = np.empty((self.height, self.width), dtype=object)

        if self.clustered:
            pass
        else:
            self.non_clustered_tile_grid(
                tile_walls_to_coordinates_map=tile_walls_to_coordinates_map
            )
