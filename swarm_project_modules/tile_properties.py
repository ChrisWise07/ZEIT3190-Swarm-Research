import fastenum


class TileColour(fastenum.Enum):
    WHITE = 0
    BLACK = 1


class WallType(fastenum.Enum):
    LEFT_WALL = 1
    RIGHT_WALL = 2
    TOP_WALL = 3
    BOTTOM_WALL = 4
