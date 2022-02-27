from pickle import TRUE
import unittest

from swarm_project_modules import Tile, TileColour, TileType


class tile_tester(unittest.TestCase):
    def setUp(self) -> None:
        self.tile = Tile(colour=TileColour.WHITE, tile_type=TileType.OPEN)

    def test_tile_return_colour_is_white(self):
        self.assertEqual(self.tile.colour.name, "WHITE")

    def test_change_tile_colour_returns_error(self):
        with self.assertRaises(AttributeError):
            self.tile.colour.name = "BLACK"

    def test_tile_return_type_open(self):
        self.assertEqual(self.tile.tile_type.name, "OPEN")

    def test_change_tile_type_returns_error(self):
        with self.assertRaises(AttributeError):
            self.tile.tile_type.name = "LEFT_WALL"

    def test_tile_occupied_returns_true_false_when_true_or_false(self):
        self.tile.set_occupied(True)
        self.assertEqual(self.tile.get_occupied(), True)
        self.tile.set_occupied(False)
        self.assertEqual(self.tile.get_occupied(), False)


if __name__ == "__main__":
    unittest.main()
