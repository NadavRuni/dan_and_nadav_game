from typing import List, Tuple
import math
from .ball import Ball


class Table:
    def __init__(self, length: float, width: float):
        self.length = length
        self.width = width

        # initialize 6 pockets (corners + middles of long sides)
        self.pockets = [
            (0, 0),  # top-left
            (length, 0),  # top-right
            (0, width),  # bottom-left
            (length, width),  # bottom-right
            (length / 2, 0),  # middle-top
            (length / 2, width)  # middle-bottom
        ]

        self.balls: List[Ball] = []
        self._setup_balls()

    def _setup_balls(self):
        """Create balls: solids (1–7), striped (9–15), black (8), white (0)."""
        # כדור לבן (נתחיל באמצע השולחן בצד אחד)
        self.balls.append(Ball(0, self.length * 0.25, self.width / 2, "white"))

        # כדור שחור במרכז
        self.balls.append(Ball(8, self.length * 0.75, self.width / 2, "black"))

        # כדורים מלאים (1–7)
        for i in range(1, 8):
            self.balls.append(Ball(i, self.length * 0.75 + (i * 2), self.width / 2, "solid"))

        # כדורים עם פסים (9–15)
        for i in range(9, 16):
            self.balls.append(Ball(i, self.length * 0.75 + ((i - 8) * 2), self.width / 2 + 5, "striped"))

    def show_balls(self):
        for ball in self.balls:
            print(ball)

    def show_pockets(self):
        print("Pockets:", self.pockets)

    def __repr__(self):
        return f"Table(length={self.length}, width={self.width}, balls={len(self.balls)})"