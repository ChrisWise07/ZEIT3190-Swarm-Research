import fastenum


class TileColour(fastenum.Enum):
    WHITE = 0
    BLACK = 1


class WallType(fastenum.Enum):
    LEFT_WALL = 0
    RIGHT_WALL = 1
    TOP_WALL = 2
    BOTTOM_WALL = 3
