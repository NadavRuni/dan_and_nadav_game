"""
Defines the main Ball object used in the game logic.

Note:
    The project contains two different 'Ball' classes. This one is used for
    the game logic calculations, while the one in 'json_models.py' is used for
    data transfer from the analysis pipeline. This duplication should be
    resolved.
"""

import math
from typing import Tuple

from const_numbers import get_ball_radius


class GameBall:
    """
    Represents a single ball in the context of the game simulation.
    """

    def __init__(
        self,
        ball_id: int,
        x_cord: float,
        y_cord: float,
        ball_type: str,
        radius: float = get_ball_radius(),
    ):
        """
        Initializes a GameBall object.

        Args:
            ball_id: Unique identifier for the ball (e.g., 0 for white,
                     8 for black, 1-7 for solids, 9-15 for striped).
            x_cord: The x-coordinate of the ball's center.
            y_cord: The y-coordinate of the ball's center.
            ball_type: A string representing the ball type, e.g., "solid",
                       "striped", "black", or "white".
            radius: The radius of the ball. Defaults to the global ball radius.
        """
        self.id = ball_id
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.type = ball_type
        self.radius = radius

    def update_position(self, new_x: float, new_y: float) -> None:
        """
        Updates the ball's position to new coordinates.

        Args:
            new_x: The new x-coordinate.
            new_y: The new y-coordinate.
        """
        self.x_cord = new_x
        self.y_cord = new_y

    def distance_to(self, other_ball: "GameBall") -> float:
        """
        Calculates the center-to-center distance to another ball.

        Args:
            other_ball: The other GameBall object.

        Returns:
            The Euclidean distance between the centers of the two balls.
        """
        delta_x = self.x_cord - other_ball.x_cord
        delta_y = self.y_cord - other_ball.y_cord
        return math.sqrt(delta_x**2 + delta_y**2)

    def get_position(self) -> Tuple[float, float]:
        """
        Returns the current (x, y) position of the ball.

        Returns:
            A tuple containing the x and y coordinates.
        """
        return (self.x_cord, self.y_cord)

    def __repr__(self) -> str:
        """
        Returns a string representation of the GameBall object.
        """
        return (
            f"GameBall(id={self.id}, type={self.type}, "
            f"pos=({self.x_cord:.2f}, {self.y_cord:.2f}), radius={self.radius:.2f})"
        )
