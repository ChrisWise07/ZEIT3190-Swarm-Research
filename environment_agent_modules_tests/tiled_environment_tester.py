import os
import sys
import unittest
from numpy import ndarray
from typing import Dict, List, Tuple

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from environment_agent_modules import (
    WallType,
    TileColour,
    navigate_tile_grid_and_call_function_over_range,
    return_ratio_of_white_to_black_tiles,
    create_nonclustered_tile_grid,
    create_clustered_inital_observation_not_useful_tile_grid,
    create_clustered_inital_observation_useful_tile_grid,
)


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = create_nonclustered_tile_grid(height=10, width=10)

    def test_tiled_enviro_creates_np_array_with_correct_dimensions(self):
        self.assertEqual(self.tiled_enviro.shape, (10, 10))

    def wall_tester(
        self,
        correct_walls: List[WallType],
        incorrect_walls: List[WallType],
        error_message: str,
        tile_grid: ndarray,
        coordinate: Tuple[int, int],
    ):
        for wall in tile_grid[coordinate]["walls"]:
            self.assertIn(
                WallType(wall),
                correct_walls,
                error_message,
            )

            self.assertNotIn(
                WallType(wall),
                incorrect_walls,
                error_message,
            )

    def colour_tester(
        self,
        colour: TileColour,
        error_message: str,
        tile_grid: ndarray,
        coordinate: Tuple[int, int],
    ):
        with self.subTest():
            self.assertEqual(
                TileColour(tile_grid[coordinate]["colour"]), colour, error_message
            )

    def test_tiled_enviro_has_correct_corners_in_correct_places(self):
        # top_left_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.TOP_WALL, WallType.LEFT_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.RIGHT_WALL],
                "error_message": "top left corner is not correct",
                "tile_grid": self.tiled_enviro,
            },
            func=self.wall_tester,
        )

        # top_right_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=(
                (self.tiled_enviro.shape[0] - 1),
                (self.tiled_enviro.shape[1]),
            ),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.TOP_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.LEFT_WALL],
                "tile_grid": self.tiled_enviro,
                "error_message": "top right corner is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_left_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.shape[0] - 1), (self.tiled_enviro.shape[0])),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.LEFT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.RIGHT_WALL],
                "tile_grid": self.tiled_enviro,
                "error_message": "bottom left corner is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_right_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.shape[0] - 1), (self.tiled_enviro.shape[0])),
            column_range=(
                (self.tiled_enviro.shape[1] - 1),
                (self.tiled_enviro.shape[1]),
            ),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.LEFT_WALL],
                "tile_grid": self.tiled_enviro,
                "error_message": "bottom right corner is not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_correct_edges_in_correct_places(self):
        # top_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=(1, (self.tiled_enviro.shape[1] - 1)),
            func_config={
                "correct_walls": [WallType.TOP_WALL],
                "incorrect_walls": [
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "tile_grid": self.tiled_enviro,
                "error_message": "top edge is not correct",
            },
            func=self.wall_tester,
        )

        # left_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.shape[0] - 1)),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.LEFT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.RIGHT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "tile_grid": self.tiled_enviro,
                "error_message": "left edge is not correct",
            },
            func=self.wall_tester,
        )

        # right_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.shape[0] - 1)),
            column_range=(
                (self.tiled_enviro.shape[1] - 1),
                (self.tiled_enviro.shape[1]),
            ),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "tile_grid": self.tiled_enviro,
                "error_message": "right edge is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.shape[0] - 1), (self.tiled_enviro.shape[0])),
            column_range=(1, (self.tiled_enviro.shape[1] - 1)),
            func_config={
                "correct_walls": [WallType.BOTTOM_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.RIGHT_WALL,
                ],
                "tile_grid": self.tiled_enviro,
                "error_message": "bottom edge is not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_no_edges_in_correct_places(self):
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.shape[0] - 1)),
            column_range=(1, (self.tiled_enviro.shape[1] - 1)),
            func_config={
                "correct_walls": [],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "tile_grid": self.tiled_enviro,
                "error_message": "open tiles are not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_approx_correct_ratio_of_white_black_tiles(self):
        new_tiled_environment = create_nonclustered_tile_grid(
            height=15, width=15, ratio_of_white_to_black_tiles=0.7
        )

        self.assertAlmostEqual(
            return_ratio_of_white_to_black_tiles(
                tile_grid=new_tiled_environment, height=15, width=15
            ),
            0.7,
            places=1,
        )

    def clustered_initial_observations_tester(
        self,
        tile_grid: ndarray,
        height: int,
        width: int,
        initial_tile_colour: TileColour,
        non_initial_tile_colour: TileColour,
        initial_tiles_row_limit: int,
        initial_tiles_column_limit: int,
    ):
        # full with initial
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, (height)),
            column_range=(0, initial_tiles_column_limit),
            func_config={
                "colour": initial_tile_colour,
                "error_message": (
                    f"clustered initial observations helpful {initial_tile_colour} tiles not correct"
                ),
                "tile_grid": tile_grid,
            },
            func=self.colour_tester,
        )

        # partial with inital
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, initial_tiles_row_limit),
            column_range=(initial_tiles_column_limit, initial_tiles_column_limit + 1),
            func_config={
                "colour": initial_tile_colour,
                "error_message": (
                    f"clustered initial observations helpful {initial_tile_colour} tiles not correct"
                ),
                "tile_grid": tile_grid,
            },
            func=self.colour_tester,
        )

        # full with non-initial
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, (height)),
            column_range=(initial_tiles_column_limit + 1, width),
            func_config={
                "colour": non_initial_tile_colour,
                "error_message": (
                    f"clustered initial observations helpful {non_initial_tile_colour} tiles not correct"
                ),
                "tile_grid": tile_grid,
            },
            func=self.colour_tester,
        )

        # partial with non-inital
        navigate_tile_grid_and_call_function_over_range(
            row_range=(initial_tiles_row_limit, (height)),
            column_range=(initial_tiles_column_limit, initial_tiles_column_limit + 1),
            func_config={
                "colour": non_initial_tile_colour,
                "error_message": (
                    f"clustered initial observations helpful {non_initial_tile_colour} tiles not correct"
                ),
                "tile_grid": tile_grid,
            },
            func=self.colour_tester,
        )

    def test_clustered_initial_observations_helpful_has_white_tiles_first(self):
        new_tiled_environment = create_clustered_inital_observation_useful_tile_grid(
            height=5,
            width=5,
            ratio_of_white_to_black_tiles=0.5,
        )

        self.clustered_initial_observations_tester(
            tile_grid=new_tiled_environment,
            height=5,
            width=5,
            initial_tile_colour=TileColour.WHITE,
            non_initial_tile_colour=TileColour.BLACK,
            initial_tiles_row_limit=2,
            initial_tiles_column_limit=2,
        )

    def test_clustered_initial_observations_not_helpful_has_black_tiles_first(self):
        new_tiled_environment = (
            create_clustered_inital_observation_not_useful_tile_grid(
                height=5,
                width=5,
                ratio_of_white_to_black_tiles=0.5,
            )
        )

        self.clustered_initial_observations_tester(
            tile_grid=new_tiled_environment,
            height=5,
            width=5,
            initial_tile_colour=TileColour.BLACK,
            non_initial_tile_colour=TileColour.WHITE,
            initial_tiles_row_limit=2,
            initial_tiles_column_limit=2,
        )

    def test_clustered_environment_has_correct_ratio_of_tiles_with_inital_obvs_useful(
        self,
    ):
        new_tiled_environment = create_clustered_inital_observation_useful_tile_grid(
            height=15,
            width=15,
            ratio_of_white_to_black_tiles=0.3,
        )

        self.assertAlmostEqual(
            return_ratio_of_white_to_black_tiles(
                tile_grid=new_tiled_environment, height=15, width=15
            ),
            0.3,
            places=1,
        )

    def test_clustered_environment_has_correct_ratio_of_tiles_with_inital_obvs_not_useful(
        self,
    ):
        new_tiled_environment = (
            create_clustered_inital_observation_not_useful_tile_grid(
                height=15,
                width=15,
                ratio_of_white_to_black_tiles=0.25,
            )
        )

        self.assertAlmostEqual(
            return_ratio_of_white_to_black_tiles(
                tile_grid=new_tiled_environment, height=15, width=15
            ),
            0.25,
            places=1,
        )


if __name__ == "__main__":
    unittest.main()
