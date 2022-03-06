from typing import Dict, List, Tuple
from .tiled_environment import TiledEnvironment
from .tile_properties import TileColour
from numpy import ndarray


def navigate_tile_grid_and_call_function_over_range(
    row_range: Tuple[int, int],
    column_range: Tuple[int, int],
    func_config: Dict,
    func,
):
    count = 0

    for row in range(*row_range):
        for column in range(*column_range):
            if func(**func_config, coordinate=(row, column)):
                count += 1

    return count


def count_of_tile_colour(tile_grid: ndarray, colour: TileColour):
    return navigate_tile_grid_and_call_function_over_range(
        row_range=(0, tile_grid.shape[0]),
        column_range=(0, tile_grid.shape[1]),
        func_config={},
        func=lambda coordinate: (tile_grid[coordinate]["colour"] == colour),
    )


def return_ratio_of_white_to_black_tiles(tiled_environment: TiledEnvironment):
    return count_of_tile_colour(
        tile_grid=tiled_environment.tile_grid, colour=TileColour.WHITE
    ) / (tiled_environment.height * tiled_environment.width)
