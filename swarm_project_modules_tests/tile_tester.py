import os
import sys
import unittest

ROOT_DIRECTORY = os.path.dirname(os.getcwd())
sys.path.append(ROOT_DIRECTORY)

from swarm_project_modules import Tile, TileColour, WallType


class tile_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tile = Tile(colour=TileColour.WHITE, walls=None)

    def test_tile_return_colour_is_white(self):
        self.assertEqual(self.tile.colour.name, "WHITE")

    def test_change_tile_colour_returns_error(self):
        with self.assertRaises(AttributeError):
            self.tile.colour.name = "BLACK"

    def test_tile_return_type_open(self):
        self.assertEqual(self.tile.walls, None)

    def test_tile_occupied_returns_true_false_when_true_or_false(self):
        self.tile.set_occupied(True)
        self.assertEqual(self.tile.get_occupied(), True)
        self.tile.set_occupied(False)
        self.assertEqual(self.tile.get_occupied(), False)

    def test_tile_with_multiple_walls_returns_correct_walls(self):
        new_tile = Tile(
            colour=TileColour.WHITE, walls=[WallType.TOP_WALL, WallType.LEFT_WALL]
        )
        self.assertIn(WallType.TOP_WALL, new_tile.walls)
        self.assertIn(WallType.LEFT_WALL, new_tile.walls)


if __name__ == "__main__":
    unittest.main()
