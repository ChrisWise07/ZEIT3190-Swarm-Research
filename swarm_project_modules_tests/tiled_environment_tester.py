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
            height=10, width=5, ratio_of_white_to_black_tiles=0.5, clustered=False
        )

    def test_tiled_enviro_returns_correct_properties(self):
        self.assertEqual(self.tiled_enviro.height, 10)
        self.assertEqual(self.tiled_enviro.width, 5)
        self.assertEqual(self.tiled_enviro.ratio_of_white_to_black_tiles, 0.5)
        self.assertEqual(self.tiled_enviro.clustered, False)

    def test_tiled_enviro_creates_np_array_with_correct_dimensions(self):
        self.assertEqual(self.tiled_enviro.tile_grid.shape, (10, 5))

    def wall_tester(
        self, walls: List[WallType], coordinates: tuple(int, int), error_message: str
    ):
        for wall in walls:
            self.assertIn(
                wall,
                self.tiled_enviro.tile_grid[coordinates].walls,
                error_message,
            )

    def test_tiled_enviro_has_correct_walls_in_correct_places(self):
        # top_left_corner
        self.wall_tester(
            walls=[WallType.LEFT_WALL, WallType.TOP_WALL],
            coordinates=(0, 0),
            error_message="top left corner is not correct",
        )


if __name__ == "__main__":
    unittest.main()
