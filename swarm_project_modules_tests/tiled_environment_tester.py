import os
import sys
import unittest

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import TiledEnvironmentClass


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = TiledEnvironmentClass(
            length=10, width=5, ratio_of_white_to_black_tiles=0.5
        )

    def test_tiled_environment_returns_correct_dimensions(self):
        assert self.tiled_enviro.length == 10
        assert self.tiled_enviro.width == 5

    def test_tiled_environment_returns_correct_ratio(self):
        assert self.tiled_enviro.ratio_of_white_to_black_tiles == 0.5


if __name__ == "__main__":
    unittest.main()
