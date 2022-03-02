from enum import Enum, unique


@unique
class TileColour(Enum):
    WHITE = 0
    BLACK = 1


@unique
class WallType(Enum):
    LEFT_WALL = 1
    RIGHT_WALL = 2
    TOP_WALL = 3
    BOTTOM_WALL = 4
