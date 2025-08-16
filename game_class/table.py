from typing import List, Tuple
from .ball import Ball


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
        self.pockets = [
            (0, 0),               # top-left
            (length, 0),          # top-right
            (0, width),           # bottom-left
            (length, width),      # bottom-right
            (length / 2, 0),      # middle-top
            (length / 2, width)   # middle-bottom
        ]

    def show_balls(self):
        for ball in self.balls:
            print(ball)

    def show_pockets(self):
        print("Pockets:", self.pockets)

    def __repr__(self):
        return f"Table(length={self.length}, width={self.width}, balls={len(self.balls)})"
