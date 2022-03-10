from math import modf
import numpy as np

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
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

    def tile_creator(
        self, 
        colour: TileColour, 
        walls: List[WallType], 
        coordinate: Tuple[int, int], 
    ):
        self.tile_grid[coordinate] = {
            "colour": colour,
            "walls": walls,
            "occupied": False,
            "id": (coordinate)
        }

    def non_clustered_environment(
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

                self.tile_creator(
                    colour=colour,
                    walls=tile_walls_to_coordinates_map.get(str((row, column))),
                    coordinate=(row, column),
                )

    def clustered_environment(
        self,
        tile_walls_to_coordinates_map: Dict[str, List[WallType]],
        intial_tile_colour: TileColour,
        non_intial_tile_colour: TileColour,
    ):
        portion_of_remaining_column, num_of_full_columns = modf(
            self.width * self.ratio_of_white_to_black_tiles
        )

        row_number_of_last_intial_tile = int(
            round(portion_of_remaining_column * self.height)
        )

        for row in range(self.height):
            for column in range(self.width):
                if column < num_of_full_columns:
                    colour = intial_tile_colour
                elif (column == num_of_full_columns) and (
                    row < row_number_of_last_intial_tile
                ):
                    colour = intial_tile_colour
                else:
                    colour = non_intial_tile_colour

                self.tile_creator(
                    colour=colour,
                    walls=tile_walls_to_coordinates_map.get(str((row, column))),
                    coordinate=(row, column),
                )

    def __post_init__(self):
        self.tile_grid = np.empty((self.height, self.width), dtype=object)

        tile_walls_to_coordinates_map = self.return_coordinate_to_walls_dict()

        if self.clustered:
            if self.initial_observations_helpful:
                self.clustered_environment(
                    tile_walls_to_coordinates_map=tile_walls_to_coordinates_map,
                    intial_tile_colour=TileColour.WHITE,
                    non_intial_tile_colour=TileColour.BLACK,
                )
            else:
                self.clustered_environment(
                    tile_walls_to_coordinates_map=tile_walls_to_coordinates_map,
                    intial_tile_colour=TileColour.BLACK,
                    non_intial_tile_colour=TileColour.WHITE,
                )
        else:
            self.non_clustered_environment(
                tile_walls_to_coordinates_map=tile_walls_to_coordinates_map
            )
