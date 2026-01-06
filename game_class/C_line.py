"""
Defines a Line class representing a line segment between two game objects.
"""

import math
from typing import Tuple, Union

from game_class.C_ball import GameBall
from game_class.C_pocket import Pocket


class Line:
    """
    Represents a line segment between two game objects (balls or pockets).
    """

    def __init__(self, obj1: Union[GameBall, Pocket], obj2: Union[GameBall, Pocket]):
        """
        Initializes a Line object.

        Args:
            obj1: The first object (GameBall or Pocket).
            obj2: The second object (GameBall or Pocket).
        """
        self.obj1 = obj1
        self.obj2 = obj2
        self.start = obj1.get_position()
        self.end = obj2.get_position()

    def length(self) -> float:
        """
        Calculates the length of the line segment.

        Returns:
            The Euclidean distance between the start and end points.
        """
        delta_x = self.start[0] - self.end[0]
        delta_y = self.start[1] - self.end[1]
        return math.sqrt(delta_x**2 + delta_y**2)

    def as_tuple(self) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Returns the line as a tuple of start and end coordinates.

        Returns:
            A tuple in the format ((x1, y1), (x2, y2)).
        """
        return self.start, self.end

    def __repr__(self) -> str:
        """
        Returns a string representation of the Line object.
        """
        return (
            f"Line(start={self.start}, end={self.end}, " f"length={self.length():.2f})"
        )
