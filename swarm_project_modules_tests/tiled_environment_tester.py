import os
import sys
import unittest
import numpy as np
from typing import List, Tuple

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import TiledEnvironmentClass, WallType


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = TiledEnvironmentClass(
            height=5, width=5, ratio_of_white_to_black_tiles=0.5, clustered=False
        )

    def test_tiled_enviro_returns_correct_properties(self):
        self.assertEqual(self.tiled_enviro.height, 5)
        self.assertEqual(self.tiled_enviro.width, 5)
        self.assertEqual(self.tiled_enviro.ratio_of_white_to_black_tiles, 0.5)
        self.assertEqual(self.tiled_enviro.clustered, False)

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
                self.tiled_enviro.tile_grid[coordinates].walls,
                error_message,
            )

        for wall in incorrect_walls:
            self.assertNotIn(
                wall,
                self.tiled_enviro.tile_grid[coordinates].walls,
                error_message,
            )

    def test_tiled_enviro_has_correct_corners_in_correct_places(self):
        # top_left_corner
        with self.subTest():
            self.wall_tester(
                correct_walls=[WallType.LEFT_WALL, WallType.TOP_WALL],
                incorrect_walls=[WallType.BOTTOM_WALL, WallType.RIGHT_WALL],
                coordinates=(0, 0),
                error_message="top left corner is not correct",
            )

        # top_right_corner
        with self.subTest():
            self.wall_tester(
                correct_walls=[WallType.RIGHT_WALL, WallType.TOP_WALL],
                incorrect_walls=[WallType.BOTTOM_WALL, WallType.LEFT_WALL],
                coordinates=(0, (self.tiled_enviro.width - 1)),
                error_message="top right corner is not correct",
            )

        # bottom_left_corner
        with self.subTest():
            self.wall_tester(
                correct_walls=[WallType.LEFT_WALL, WallType.BOTTOM_WALL],
                incorrect_walls=[WallType.TOP_WALL, WallType.RIGHT_WALL],
                coordinates=((self.tiled_enviro.height - 1), 0),
                error_message="bottom left corner is not correct",
            )

        # bottom_right_corner
        with self.subTest():
            self.wall_tester(
                correct_walls=[WallType.RIGHT_WALL, WallType.BOTTOM_WALL],
                incorrect_walls=[WallType.TOP_WALL, WallType.LEFT_WALL],
                coordinates=(
                    (self.tiled_enviro.height - 1),
                    (self.tiled_enviro.width - 1),
                ),
                error_message="bottom right corner is not correct",
            )

    def test_tiled_enviro_has_correct_edges_in_correct_places(self):
        # top_edge
        for i in range(1, (self.tiled_enviro.width - 1)):
            with self.subTest():
                self.wall_tester(
                    correct_walls=[WallType.TOP_WALL],
                    incorrect_walls=[
                        WallType.BOTTOM_WALL,
                        WallType.RIGHT_WALL,
                        WallType.LEFT_WALL,
                    ],
                    coordinates=(0, i),
                    error_message="top edge is not correct",
                )
        # left_edge
        for i in range(1, (self.tiled_enviro.height - 1)):
            with self.subTest():
                self.wall_tester(
                    correct_walls=[WallType.LEFT_WALL],
                    incorrect_walls=[
                        WallType.TOP_WALL,
                        WallType.RIGHT_WALL,
                        WallType.BOTTOM_WALL,
                    ],
                    coordinates=(i, 0),
                    error_message="left edge is not correct",
                )
        # right_edge
        for i in range(1, (self.tiled_enviro.height - 1)):
            with self.subTest():
                self.wall_tester(
                    correct_walls=[WallType.RIGHT_WALL],
                    incorrect_walls=[
                        WallType.TOP_WALL,
                        WallType.LEFT_WALL,
                        WallType.BOTTOM_WALL,
                    ],
                    coordinates=(i, (self.tiled_enviro.width - 1)),
                    error_message="right edge is not correct",
                )
        # bottom_edge
        for i in range(1, (self.tiled_enviro.width - 1)):
            with self.subTest():
                self.wall_tester(
                    correct_walls=[WallType.BOTTOM_WALL],
                    incorrect_walls=[
                        WallType.TOP_WALL,
                        WallType.LEFT_WALL,
                        WallType.RIGHT_WALL,
                    ],
                    coordinates=((self.tiled_enviro.height - 1), i),
                    error_message="bottom edge is not correct",
                )

    def test_tiled_enviro_has_no_edges_in_correct_places(self):
        for row in range(1, (self.tiled_enviro.height - 1)):
            for column in range(1, (self.tiled_enviro.width - 1)):
                with self.subTest():
                    self.wall_tester(
                        correct_walls=[],
                        incorrect_walls=[
                            WallType.TOP_WALL,
                            WallType.LEFT_WALL,
                            WallType.RIGHT_WALL,
                            WallType.BOTTOM_WALL,
                        ],
                        coordinates=(row, column),
                        error_message="open tiles are not correct",
                    )


if __name__ == "__main__":
    unittest.main()
