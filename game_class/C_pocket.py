from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class Pocket:
    id: int
    center: Tuple[float, float]
    radius: float
    pocket_img_cordinates_on_table: Optional[Tuple[int, int]] = None
    location: Optional[str] = None
    pocket_img_path: Optional[str] = None

    def position(self) -> Tuple[float, float]:
        """Return (x, y) position of the pocket."""
        return self.center

    def __repr__(self):
        return (
            f"Pocket(id={self.id}, pos=({self.center[0]}, {self.center[1]}), "
            f"radius={self.radius}, location={self.location})"
        )