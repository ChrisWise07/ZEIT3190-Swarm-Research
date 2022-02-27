from dataclasses import dataclass, field
from .tile_properties import TileColour, TileType


@dataclass(repr=False, eq=False)
class Tile:
    colour: TileColour
    tile_type: TileType
    occupied: bool = field(init=False)

    def set_occupied(self, occupy_status: bool) -> None:
        self.occupied = occupy_status

    def get_occupied(self) -> bool:
        return self.occupied
