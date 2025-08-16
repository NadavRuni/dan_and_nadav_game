from const_numbers import CORNER_POCKET_RADIUS, SIDE_POCKET_RADIUS
from typing import Tuple


class Pocket:
    def __init__(self, pocket_id: int, x_cord: float, y_cord: float, is_corner: bool):
        """
        Initialize a Pocket (hole) object.

        Args:
            pocket_id (int): Unique identifier for the pocket (0–5)
            x_cord (float): X coordinate of the pocket
            y_cord (float): Y coordinate of the pocket
            is_corner (bool): True if pocket is a corner pocket
        """
        self.id = pocket_id
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.is_corner = is_corner
        self.radius = CORNER_POCKET_RADIUS if is_corner else SIDE_POCKET_RADIUS

    def position(self) -> Tuple[float, float]:
        """Return (x, y) position of the pocket."""
        return (self.x_cord, self.y_cord)

    def __repr__(self):
        return (
            f"Pocket(id={self.id}, pos=({self.x_cord}, {self.y_cord}), "
            f"radius={self.radius}, corner={self.is_corner})"
        )
