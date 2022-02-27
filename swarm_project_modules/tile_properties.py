from enum import Enum, unique


@unique
class TileColour(Enum):
    WHITE = 0
    BLACK = 1


@unique
class TileType(Enum):
    OPEN = 0
    LEFT_WALL = 1
    RIGHT_WALL = 2
    LEFT_CORNER = 3
    RIGHT_CORNER = 4
