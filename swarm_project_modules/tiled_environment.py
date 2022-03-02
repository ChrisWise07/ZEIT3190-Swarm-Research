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
    tile_corners_to_coordinates_map: Dict[Tuple[int, int], List[WallType]] = field(
        init=False
    )

    def return_tile_with_walls_based_on_coordinates(
        self, coordinates: Tuple[int, int], colour: TileColour
    ) -> Tile:
        return Tile(
            colour=colour,
            walls=self.tile_corners_to_coordinates_map[str(coordinates)],
        )
        """
        if str(coordinates) in self.tile_corners_to_coordinates_map:
            return Tile(
                colour=colour,
                walls=self.tile_corners_to_coordinates_map[str(coordinates)],
            )
        else:
            return Tile(colour=colour, walls=None)
        """

    def __post_init__(self):
        self.tile_corners_to_coordinates_map = RegexDict(
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
                # top edge
                f"(0, [1-{self.width - 2}])": [WallType.TOP_WALL],
            }
        )

        self.tile_grid = np.empty((self.height, self.width), dtype=object)

        for row in range(self.height):
            for column in range(self.width):
                self.tile_grid[
                    row, column
                ] = self.return_tile_with_walls_based_on_coordinates(
                    coordinates=(row, column), colour=TileColour.WHITE
                )
