"""
Defines the Pocket object used in the game logic.
"""

from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Pocket:
    """
    Represents a pocket on the pool table.
    """

    id: int
    center: Tuple[float, float]
    radius: float
    pocket_img_cordinates_on_table: Optional[Tuple[int, int]] = None
    location: Optional[str] = None
    pocket_img_path: Optional[str] = None

    def get_position(self) -> Tuple[float, float]:
        """
        Returns the (x, y) position of the pocket's center.

        Returns:
            A tuple containing the x and y coordinates.
        """
        return self.center

    def __repr__(self) -> str:
        """
        Returns a string representation of the Pocket object.
        """
        return (
            f"Pocket(id={self.id}, pos=({self.center[0]:.2f}, {self.center[1]:.2f}), "
            f"radius={self.radius:.2f}, location={self.location})"
        )
