from dataclasses import dataclass


@dataclass(repr=False, eq=False)
class TiledEnvironmentClass:
    length: int
    width: int
    ratio_of_white_to_black_tiles: float
