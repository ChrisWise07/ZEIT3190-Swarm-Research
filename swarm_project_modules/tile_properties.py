import fastenum


class TileColour(fastenum.Enum):
    WHITE = 0
    BLACK = 1


class WallType(fastenum.Enum):
    TOP_WALL = 0
    RIGHT_WALL = 1
    BOTTOM_WALL = 2
    LEFT_WALL = 3
