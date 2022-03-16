import fastenum


class Direction(fastenum.Enum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


class Turn(fastenum.Enum):
    RIGHT = 1
    LEFT = -1


class RelativePosition(fastenum.Enum):
    FRONT = 0
    LEFT = 1
    BEHIND = 2
    RIGHT = 3


class ObjectType(fastenum.Enum):
    NONE = 0
    WALL = 1
    CORNER = 2
    AGENT = 3


class RelativeMotion(fastenum.Enum):
    ESCAPING = 0
    APPROACHING = 1
