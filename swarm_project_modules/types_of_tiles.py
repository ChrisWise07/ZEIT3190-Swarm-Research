from enum import Enum, unique
from .tile import Tile
from .tile_properties import TileColour, WallType


@unique
class TileType(Enum):
    TOP_LEFT_WALL = Tile(colour=Ti)
    RIGHT_WALL = 2
    TOP_WALL = 3
    BOTTOM_WALL = 4
