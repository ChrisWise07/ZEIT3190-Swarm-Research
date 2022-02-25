from dataclasses import dataclass


@dataclass(repr=False, eq=False)
class tiled_environment_class:
    length: int
    width: int
    ratio_of_white_to_black_tiles: float
