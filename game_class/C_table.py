from typing import List
from .C_ball import Ball
from game_class.C_pocket import Pocket
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
        if not get_use_predicted_pockets():
            self.pockets: List[Pocket] = [
                Pocket(
                    id=3,
                    center=(
                        0 + get_min_distance_from_pocket(),
                        0 + get_min_distance_from_pocket(),
                    ),
                    radius=get_corner_pocket_radius(),
                    pocket_img_cordinates_on_table=(
                        0 + get_min_distance_from_pocket(),
                        0 + get_min_distance_from_pocket(),
                    ),
                    location="  TL",
                ),  # TOP-left
                Pocket(
                    id=2,
                    center=(
                        length - get_min_distance_from_pocket(),
                        0 + get_min_distance_from_pocket(),
                    ),
                    radius=get_corner_pocket_radius(),
                    pocket_img_cordinates_on_table=(
                        length - get_min_distance_from_pocket(),
                        0 + get_min_distance_from_pocket(),
                    ),
                    location="TR",
                ),  # bottom-right
                Pocket(
                    id=1,
                    center=(
                        length - get_min_distance_from_pocket(),
                        width - get_min_distance_from_pocket(),
                    ),
                    radius=get_corner_pocket_radius(),
                    pocket_img_cordinates_on_table=(
                        length - get_min_distance_from_pocket(),
                        width - get_min_distance_from_pocket(),
                    ),
                    location="BR",
                ),  # buttom-right
                Pocket(
                    id=0,
                    center=(
                        0 + get_min_distance_from_pocket(),
                        width - get_min_distance_from_pocket(),
                    ),
                    radius=get_corner_pocket_radius(),
                    pocket_img_cordinates_on_table=(
                        0 + get_min_distance_from_pocket(),
                        width - get_min_distance_from_pocket(),
                    ),
                    location="BL",
                ),  # top-left
            Pocket(id=4, center=(length / 2, width), radius=get_side_pocket_radius(), pocket_img_cordinates_on_table=(length / 2, width), location="BM"),  # middle-bottom
            Pocket(id=5, center=(length / 2, 0), radius=get_side_pocket_radius(), pocket_img_cordinates_on_table=(length / 2, 0), location="TM"),  # middle-top
            ]
        else:
            print ("Using predicted pockets")
            self.pockets: List[Pocket] = get_detected_pockets()    
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
