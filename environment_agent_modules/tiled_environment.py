from math import modf
import numpy as np

from typing import Dict, List, Tuple
from .tile_properties import TileColour, WallType
from .regex_dictionary import RegexDict


def return_coordinate_to_walls_dict(width: int, height: int) -> RegexDict:
    return RegexDict(
        {
            # top left corner
            "(0, 0)": [WallType.TOP_WALL.value, WallType.LEFT_WALL.value],
            # top right corner
            f"(0, {(width - 1)})": [WallType.TOP_WALL.value, WallType.RIGHT_WALL.value],
            # bottom left corner
            f"({(height - 1)}, 0)": [
                WallType.BOTTOM_WALL.value,
                WallType.LEFT_WALL.value,
            ],
            # bottom right corner
            f"({(height - 1)}, {(width - 1)})": [
                WallType.BOTTOM_WALL.value,
                WallType.RIGHT_WALL.value,
            ],
            # top_edge
            "(0, .)": [WallType.TOP_WALL.value],
            # left_wall
            "(., 0)": [WallType.LEFT_WALL.value],
            # right_wall
            f"(., {(width - 1)})": [WallType.RIGHT_WALL.value],
            # bottom_edge
            f"({(height - 1)}, .)": [WallType.BOTTOM_WALL.value],
            # all other tiles
            "(., .)": [],
        }
    )


def tile_creator(
    colour: TileColour,
    walls: List[int],
    coordinate: Tuple[int, int],
):
    return {
        "colour": colour,
        "walls": walls,
        "occupied": False,
        "id": (coordinate),
    }


def non_clustered_environment(
    width: int,
    height: int,
    ratio_of_white_to_black_tiles: float,
    tile_grid: np.ndarray,
    tile_walls_to_coordinates_map: Dict[str, List[int]],
):
    random_numbers_between_0_1 = np.random.rand(height, width)

    for row in range(height):
        for column in range(width):
            if (
                random_numbers_between_0_1[(row, column)]
                < ratio_of_white_to_black_tiles
            ):
                colour = TileColour.WHITE.value
            else:
                colour = TileColour.BLACK.value

            tile_grid[(row, column)] = tile_creator(
                colour=colour,
                walls=tile_walls_to_coordinates_map.get(str((row, column))),
                coordinate=(row, column),
            )

    return tile_grid


def clustered_environment(
    width: int,
    height: int,
    ratio_of_inital_to_non_inital_tile: float,
    tile_grid: np.ndarray,
    tile_walls_to_coordinates_map: Dict[str, List[int]],
    intial_tile_colour: int,
    non_intial_tile_colour: int,
):
    portion_of_remaining_column, num_of_full_columns = modf(
        width * ratio_of_inital_to_non_inital_tile
    )

    row_number_of_last_intial_tile = int(round(portion_of_remaining_column * height))

    for row in range(height):
        for column in range(width):
            if column < num_of_full_columns:
                colour = intial_tile_colour
            elif (column == num_of_full_columns) and (
                row < row_number_of_last_intial_tile
            ):
                colour = intial_tile_colour
            else:
                colour = non_intial_tile_colour

            tile_grid[(row, column)] = tile_creator(
                colour=colour,
                walls=tile_walls_to_coordinates_map.get(str((row, column))),
                coordinate=(row, column),
            )

    return tile_grid


def create_clustered_inital_observation_useful_tile_grid(
    width: int,
    height: int,
    ratio_of_white_to_black_tiles: float = 0.5,
) -> np.ndarray:
    return clustered_environment(
        width=width,
        height=height,
        ratio_of_inital_to_non_inital_tile=ratio_of_white_to_black_tiles,
        tile_grid=np.empty((height, width), dtype=object),
        tile_walls_to_coordinates_map=return_coordinate_to_walls_dict(
            width=width, height=height
        ),
        intial_tile_colour=TileColour.WHITE.value,
        non_intial_tile_colour=TileColour.BLACK.value,
    )


def create_clustered_inital_observation_not_useful_tile_grid(
    width: int,
    height: int,
    ratio_of_white_to_black_tiles: float = 0.5,
) -> np.ndarray:
    return clustered_environment(
        width=width,
        height=height,
        ratio_of_inital_to_non_inital_tile=(1 - ratio_of_white_to_black_tiles),
        tile_grid=np.empty((height, width), dtype=object),
        tile_walls_to_coordinates_map=return_coordinate_to_walls_dict(
            width=width, height=height
        ),
        intial_tile_colour=TileColour.BLACK.value,
        non_intial_tile_colour=TileColour.WHITE.value,
    )


def create_nonclustered_tile_grid(
    width: int,
    height: int,
    ratio_of_white_to_black_tiles: float = 0.5,
) -> np.ndarray:
    return non_clustered_environment(
        width=width,
        height=height,
        ratio_of_white_to_black_tiles=ratio_of_white_to_black_tiles,
        tile_grid=np.empty((height, width), dtype=object),
        tile_walls_to_coordinates_map=return_coordinate_to_walls_dict(
            width=width, height=height
        ),
    )
