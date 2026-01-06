"""
Defines the Table class, which represents the game state.
"""

from typing import List

from const_numbers import (
    get_corner_pocket_radius,
    get_detected_pockets,
    get_min_distance_from_pocket,
    get_side_pocket_radius,
    get_use_predicted_pockets,
)
from game_class.C_ball import GameBall
from game_class.C_pocket import Pocket


class Table:
    """
    Represents the pool table, including its dimensions, balls, and pockets.
    """

    def __init__(self, length: float, width: float, balls: List[GameBall]):
        """
        Initializes the Table object.

        Note:
            The pocket initialization logic is critically flawed. It depends on
            a global state flag (`get_use_predicted_pockets`) to decide whether
            to use hardcoded pocket locations or a globally stored list of
            predicted pockets. The pocket list should be passed as an argument
            to this constructor.

        Args:
            length: The length of the table.
            width: The width of the table.
            balls: A list of GameBall objects to place on the table.
        """
        self.length = length
        self.width = width
        self.balls = balls
        self.pockets: List[Pocket] = []

        if get_use_predicted_pockets():
            print("Using predicted pockets from global state.")
            self.pockets = get_detected_pockets() or []
        else:
            self._initialize_default_pockets()

    def _initialize_default_pockets(self) -> None:
        """Initializes the 6 standard pockets with hardcoded positions."""
        # This hardcoded logic is inflexible and difficult to maintain.
        pocket_margin = get_min_distance_from_pocket()
        self.pockets = [
            Pocket(
                id=3,
                center=(pocket_margin, pocket_margin),
                radius=get_corner_pocket_radius(),
                location="TL",
            ),
            Pocket(
                id=2,
                center=(self.length - pocket_margin, pocket_margin),
                radius=get_corner_pocket_radius(),
                location="TR",
            ),
            Pocket(
                id=1,
                center=(self.length - pocket_margin, self.width - pocket_margin),
                radius=get_corner_pocket_radius(),
                location="BR",
            ),
            Pocket(
                id=0,
                center=(pocket_margin, self.width - pocket_margin),
                radius=get_corner_pocket_radius(),
                location="BL",
            ),
            Pocket(
                id=4,
                center=(self.length / 2, self.width),
                radius=get_side_pocket_radius(),
                location="BM",
            ),
            Pocket(
                id=5,
                center=(self.length / 2, 0),
                radius=get_side_pocket_radius(),
                location="TM",
            ),
        ]

    def show_balls(self) -> None:
        """Prints a representation of each ball on the table."""
        for ball in self.balls:
            print(ball)

    def show_pockets(self) -> None:
        """Prints a representation of each pocket on the table."""
        for pocket in self.pockets:
            print(pocket)

    def get_length(self) -> float:
        """Returns the length of the table."""
        return self.length

    def get_width(self) -> float:
        """Returns the width of the table."""
        return self.width

    def get_pockets(self) -> List[Pocket]:
        """Returns the list of pockets on the table."""
        return self.pockets

    def get_balls(self) -> List[GameBall]:
        """Returns the list of all balls on the table."""
        return self.balls

    def get_solid_balls(self) -> List[GameBall]:
        """Returns a list of only the solid balls."""
        return [ball for ball in self.balls if ball.type == "solid"]

    def get_striped_balls(self) -> List[GameBall]:
        """Returns a list of only the striped balls."""
        return [ball for ball in self.balls if ball.type == "striped"]

    def get_black_ball(self) -> List[GameBall]:
        """Returns a list containing the black ball, if present."""
        return [ball for ball in self.balls if ball.type == "black"]

    def __repr__(self) -> str:
        """Returns a string representation of the Table object."""
        return (
            f"Table(length={self.length}, width={self.width}, "
            f"balls={len(self.balls)}, pockets={len(self.pockets)})"
        )
