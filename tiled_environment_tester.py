import unittest

from swarm_project_files import tiled_environment_class


class tiled_environment_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tiled_enviro = tiled_environment_class(
            length=10, width=5, ratio_of_white_to_black_tiles=0.5
        )

    def test_tiled_environment_object_is_created(self):
        assert (bool(self.tiled_enviro), True)

    def test_tiled_environment_returns_correct_dimensions(self):
        assert (self.tiled_enviro.length, 10)
        assert (self.tiled_enviro.width, 5)

    def test_tiled_environment_returns_correct_ratio(self):
        assert (self.tiled_enviro.ratio_of_white_to_black_tiles, 0.5)


if __name__ == "__main__":
    unittest.main()
