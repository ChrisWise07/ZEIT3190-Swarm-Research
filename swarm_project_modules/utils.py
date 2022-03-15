from typing import Dict, Tuple
from numpy import ndarray
from .tile_properties import TileColour
from .swarm_agent_enums import ObjectType


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


def return_ratio_of_white_to_black_tiles(tile_grid: ndarray, height: int, width: int):
    return count_of_tile_colour(tile_grid=tile_grid, colour=TileColour.WHITE) / (
        height * width
    )


def validate_cell(new_cell: Tuple[int, int], grid_shape: Tuple[int, int]) -> bool:
    return all(
        [
            (row_or_col >= 0 and row_or_col < dimension)
            for row_or_col, dimension in zip(new_cell, grid_shape)
        ]
    )


def get_object_type_based_on_num_wall(num_of_walls: int) -> ObjectType:
    return {0: ObjectType.NONE, 1: ObjectType.WALL, 2: ObjectType.CORNER}[num_of_walls]
