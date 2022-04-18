import fastenum


class TileColour(fastenum.Enum):
    WHITE = 1
    BLACK = 0


class WallType(fastenum.Enum):
    TOP_WALL = 0
    RIGHT_WALL = 1
    BOTTOM_WALL = 2
    LEFT_WALL = 3
