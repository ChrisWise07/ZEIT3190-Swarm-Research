import os
import sys
import unittest
import numpy as np
from typing import Dict, List, Tuple

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import TiledEnvironmentClass, WallType


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = TiledEnvironmentClass(
            height=5, width=5, ratio_of_white_to_black_tiles=0.5, clustered=False
        )

    def test_tiled_enviro_creates_np_array_with_correct_dimensions(self):
        self.assertEqual(self.tiled_enviro.tile_grid.shape, (5, 5))

    def wall_tester(
        self,
        correct_walls: List[WallType],
        incorrect_walls: List[WallType],
        coordinates: Tuple[int, int],
        error_message: str,
    ):
        for wall in correct_walls:
            self.assertIn(
                wall,
                self.tiled_enviro.tile_grid[coordinates]["walls"],
                error_message,
            )

        for wall in incorrect_walls:
            self.assertNotIn(
                wall,
                self.tiled_enviro.tile_grid[coordinates]["walls"],
                error_message,
            )

    def navigate_tile_grid_and_call_wall_tester_with_coordinate(
        self,
        row_range: Tuple[int, int],
        column_range: Tuple[int, int],
        wall_test_config: Dict,
    ):
        for row in range(*row_range):
            for column in range(*column_range):
                with self.subTest():
                    self.wall_tester(
                        correct_walls=wall_test_config["correct_walls"],
                        incorrect_walls=wall_test_config["incorrect_walls"],
                        coordinates=(row, column),
                        error_message=wall_test_config["error_message"],
                    )

    def test_tiled_enviro_has_correct_corners_in_correct_places(self):
        # top_left_corner
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(0, 1),
            column_range=(0, 1),
            wall_test_config={
                "correct_walls": [WallType.LEFT_WALL, WallType.TOP_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.RIGHT_WALL],
                "error_message": "top left corner is not correct",
            },
        )

        # top_right_corner
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(0, 1),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            wall_test_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.TOP_WALL],
                "incorrect_walls": [WallType.BOTTOM_WALL, WallType.LEFT_WALL],
                "error_message": "top right corner is not correct",
            },
        )

        # bottom_left_corner
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=(0, 1),
            wall_test_config={
                "correct_walls": [WallType.LEFT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.RIGHT_WALL],
                "error_message": "bottom left corner is not correct",
            },
        )

        # bottom_right_corner
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            wall_test_config={
                "correct_walls": [WallType.RIGHT_WALL, WallType.BOTTOM_WALL],
                "incorrect_walls": [WallType.TOP_WALL, WallType.LEFT_WALL],
                "error_message": "bottom right corner is not correct",
            },
        )

    def test_tiled_enviro_has_correct_edges_in_correct_places(self):
        # top_edge
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(0, 1),
            column_range=(1, (self.tiled_enviro.width - 1)),
            wall_test_config={
                "correct_walls": [WallType.TOP_WALL],
                "incorrect_walls": [
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "error_message": "top edge is not correct",
            },
        )

        # left_edge
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=(0, 1),
            wall_test_config={
                "correct_walls": [WallType.LEFT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.RIGHT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "error_message": "left edge is not correct",
            },
        )

        # right_edge
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=((self.tiled_enviro.width - 1), (self.tiled_enviro.width)),
            wall_test_config={
                "correct_walls": [WallType.RIGHT_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.BOTTOM_WALL,
                ],
                "error_message": "right edge is not correct",
            },
        )

        # bottom_edge
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=((self.tiled_enviro.height - 1), (self.tiled_enviro.height)),
            column_range=(1, (self.tiled_enviro.width - 1)),
            wall_test_config={
                "correct_walls": [WallType.BOTTOM_WALL],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.LEFT_WALL,
                    WallType.RIGHT_WALL,
                ],
                "error_message": "bottom edge is not correct",
            },
        )

    def test_tiled_enviro_has_no_edges_in_correct_places(self):
        self.navigate_tile_grid_and_call_wall_tester_with_coordinate(
            row_range=(1, (self.tiled_enviro.height - 1)),
            column_range=(1, (self.tiled_enviro.width - 1)),
            wall_test_config={
                "correct_walls": [],
                "incorrect_walls": [
                    WallType.TOP_WALL,
                    WallType.BOTTOM_WALL,
                    WallType.RIGHT_WALL,
                    WallType.LEFT_WALL,
                ],
                "error_message": "open tiles are not correct",
            },
        )


if __name__ == "__main__":
    unittest.main()
