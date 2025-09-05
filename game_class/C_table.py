from typing import List
from .C_ball import Ball
from .C_pocket import Pocket
from const_numbers import *


class Table:
    def __init__(self, length: float, width: float, balls: List[Ball]):
        """
        Table constructor.

        Args:
            length (float): Length of the table
            width (float): Width of the table
            balls (List[Ball]): Pre-initialized balls to place on the table
        """
        self.length = length
        self.width = width
        self.balls = balls

        # initialize 6 pockets (corners + middles of long sides)
        self.pockets: List[Pocket] = [
            Pocket(
                0, 0 + ADD_TO_POCKET, 0 + ADD_TO_POCKET, is_corner=True
            ),  # bottom-left
            Pocket(
                1, length - ADD_TO_POCKET, 0 + ADD_TO_POCKET, is_corner=True
            ),  # bottom-right
            Pocket(
                2, length - ADD_TO_POCKET, width - ADD_TO_POCKET, is_corner=True
            ),  # top-right
            Pocket(
                3, 0 + ADD_TO_POCKET, width - ADD_TO_POCKET, is_corner=True
            ),  # top-left
            Pocket(4, length / 2, 0, is_corner=False),  # middle-bottom
            Pocket(5, length / 2, width, is_corner=False),  # middle-top
        ]

    def show_balls(self):
        for ball in self.balls:
            print(ball)

    def get_length(self) -> float:
        return self.length

    def get_width(self) -> float:
        return self.width

    def get_pockets(self) -> List[Pocket]:
        return self.pockets

    def get_balls(self) -> List[Ball]:
        return self.balls

    def get_solid(self) -> List[Ball]:
        return [ball for ball in self.balls if ball.type == "solid"]

    def get_striped(self) -> List[Ball]:
        return [ball for ball in self.balls if ball.type == "striped"]

    def get_black(self) -> List[Ball]:
        return [ball for ball in self.balls if ball.type == "black"]

    def show_pockets(self):
        for pocket in self.pockets:
            print(pocket)

    def __repr__(self):
        return f"Table(length={self.length}, width={self.width}, balls={len(self.balls)}, pockets={len(self.pockets)})"
