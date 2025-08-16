import math
from typing import Tuple, Union
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket


class Line:
    def __init__(self, obj1: Union[Ball, Pocket], obj2: Union[Ball, Pocket]):
        """
        Initialize a Line object between two objects (Ball or Pocket).

        Args:
            obj1 (Ball|Pocket): First object
            obj2 (Ball|Pocket): Second object
        """
        self.obj1 = obj1
        self.obj2 = obj2
        self.start = obj1.position()
        self.end = obj2.position()

    def length(self) -> float:
        """Return the distance between the two objects (length of the line)."""
        dx = self.start[0] - self.end[0]
        dy = self.start[1] - self.end[1]
        return math.sqrt(dx**2 + dy**2)

    def as_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """Return the line as ((x1, y1), (x2, y2))"""
        return self.start, self.end

    def __repr__(self):
        return f"Line(start={self.start}, end={self.end}, length={self.length():.2f})"
