import math
from typing import Tuple
from const_numbers import *

class Ball:
    def __init__(self, ball_id: int, x_cord: float, y_cord: float, ball_type: str, radius: float = BALL_RADIUS):
        """
        Initialize a Ball object.

        Args:
            ball_id (int): Unique identifier for the ball (0=white, 8=black, 1-7=solids, 9-15=striped)
            x_cord (float): X coordinate of the ball
            y_cord (float): Y coordinate of the ball
            ball_type (str): "solid", "striped", "black", or "white"
            radius (float): Radius of the ball (default = BALL_RADIUS)
        """
        self.id = ball_id
        self.x_cord = x_cord
        self.y_cord = y_cord
        self.type = ball_type
        self.radius = radius

    def update_position(self, new_x: float, new_y: float):
        """Update the ball's position."""
        self.x_cord = new_x
        self.y_cord = new_y

    def distance_to(self, other: "Ball") -> float:
        """Calculate the distance to another ball (center to center)."""
        dx = self.x_cord - other.x_cord
        dy = self.y_cord - other.y_cord
        return math.sqrt(dx**2 + dy**2)

    def position(self) -> Tuple[float, float]:
        """Return (x,y) position of the ball."""
        return (self.x_cord, self.y_cord)

    def __repr__(self):
        return (f"Ball(id={self.id}, type={self.type}, "
                f"pos=({self.x_cord}, {self.y_cord}), radius={self.radius})")