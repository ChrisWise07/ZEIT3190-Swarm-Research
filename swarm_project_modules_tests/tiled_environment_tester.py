import os
import sys
import unittest
from numpy import ndarray
from typing import Dict, List, Tuple

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import (
    TiledEnvironment,
    WallType,
    TileColour,
    navigate_tile_grid_and_call_function_over_range,
    return_ratio_of_white_to_black_tiles,
)


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = TiledEnvironment(
            height=10, width=10, ratio_of_white_to_black_tiles=0.5, clustered=False
        )

    def test_tiled_enviro_creates_np_array_with_correct_dimensions(self):
        self.assertEqual(self.tiled_enviro.tile_grid.shape, (10, 10))

    def wall_tester(
        self,
        correct_walls: List[WallType],
        incorrect_walls: List[WallType],
        error_message: str,
        tile_grid: ndarray,
        coordinate: Tuple[int, int],
    ):
        with self.subTest():
            for wall in correct_walls:
                self.assertIn(
                    wall,
                    tile_grid[coordinate]["walls"],
                    error_message,
                )

            for wall in incorrect_walls:
                self.assertNotIn(
                    wall,
                    tile_grid[coordinate]["walls"],
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
            self.assertEqual(tile_grid[coordinate]["colour"], colour, error_message)

    def test_tiled_enviro_has_correct_corners_in_correct_places(self):
        # top_left_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.LEFT_WALL, WallType.TOP_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.RIGHT_WALL],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "top left corner is not correct",
            },
            func=self.wall_tester,
        )

        # top_right_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.TOP_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.LEFT_WALL],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "top right corner is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_left_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.LEFT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.RIGHT_WALL],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "bottom left corner is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_right_corner
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.LEFT_WALL],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "bottom right corner is not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_correct_edges_in_correct_places(self):
        # top_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, 1),
            column_range=(1, (self.tiled_enviro.width - 1)),
            func_config={
                "correct_walls": [WallType.TOP_WALL],
                "incorrect_walls": [
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "top edge is not correct",
            },
            func=self.wall_tester,
        )

        # left_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=(0, 1),
            func_config={
                "correct_walls": [WallType.LEFT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.RIGHT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "left edge is not correct",
            },
            func=self.wall_tester,
        )

        # right_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            func_config={
                "correct_walls": [WallType.RIGHT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "right edge is not correct",
            },
            func=self.wall_tester,
        )

        # bottom_edge
        navigate_tile_grid_and_call_function_over_range(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=(1, (self.tiled_enviro.width - 1)),
            func_config={
                "correct_walls": [WallType.BOTTOM_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.RIGHT_WALL,
                ],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "bottom edge is not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_no_edges_in_correct_places(self):
        navigate_tile_grid_and_call_function_over_range(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=(1, (self.tiled_enviro.width - 1)),
            func_config={
                "correct_walls": [],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "tile_grid": self.tiled_enviro.tile_grid,
                "error_message": "open tiles are not correct",
            },
            func=self.wall_tester,
        )

    def test_tiled_enviro_has_approx_correct_ratio_of_white_black_tiles(self):
        new_tiled_environment = TiledEnvironment(
            height=15, width=15, ratio_of_white_to_black_tiles=0.7
        )

        self.assertAlmostEqual(
            return_ratio_of_white_to_black_tiles(
                tiled_environment=new_tiled_environment
            ),
            0.7,
            places=1,
        )

    def test_clustered_initial_observations_helpful_has_white_tiles_first(self):
        new_tiled_environment = TiledEnvironment(
            height=5,
            width=5,
            ratio_of_white_to_black_tiles=0.5,
            clustered=True,
            initial_observations_helpful=True,
        )

        # initial_white_tiles
        navigate_tile_grid_and_call_function_over_range(
            row_range=(0, (self.tiled_enviro.height - 1)),
            column_range=(0, 2),
            func_config={
                "colour": TileColour.WHITE,
                "error_message": "clustered initial observations helpful white tiles not correct",
                "tile_grid": new_tiled_environment.tile_grid,
            },
            func=self.colour_tester,
        )


if __name__ == "__main__":
    unittest.main()
