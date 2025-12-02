"""
Represents and evaluates a direct shot from the cue ball to a target ball.
"""

import math
from typing import Optional, Tuple, List

from const_numbers import (
    FORSE_WALL_SHOT,
    NOT_FREE_SHOT,
    get_max_white_to_target_distance,
)
from game_class.C_ball import GameBall
from game_class.C_calc import Calculations
from game_class.C_pocket import Pocket
from game_class.C_table import Table


class BestShot:
    """
    Analyzes and scores a single direct shot from a cue ball to a target.

    The constructor performs all calculations to determine if a valid shot to
    any pocket exists. If so, it stores the details and score of the best one found.

    Attributes:
        white (GameBall): The cue ball.
        target (GameBall): The target ball.
        table (Table): The table object containing all game state.
        pocket (Optional[Pocket]): The best pocket for the shot, if one exists.
        valid (bool): True if a valid shot was found, False otherwise.
        score (float): The final calculated score for the shot.
    """

    def __init__(self, white: GameBall, target: GameBall, table: Table):
        """
        Initializes and evaluates the direct shot.
        """
        self.white = white
        self.target = target
        self.table = table
        self.valid = False

        # Find the best pocket and angle for this white->target combination
        calc = Calculations(white, target, table)
        best_pocket_id, best_angle = calc.find_best_shot_angle()

        # Check if a valid shot was found
        if (
            best_pocket_id == NOT_FREE_SHOT
            or best_angle == float("inf")
            or FORSE_WALL_SHOT
        ):
            self._set_as_invalid()
            return

        if abs(best_angle) > 75:
            self._set_as_invalid()
            return

        # A valid shot exists, so we calculate its properties and score
        self.valid = True
        self.pocket: Optional[Pocket] = next(
            (p for p in table.pockets if p.id == best_pocket_id), None
        )
        self.angle: float = best_angle

        self.dist_target_to_pocket = math.hypot(
            self.pocket.center[0] - target.x_cord,
            self.pocket.center[1] - target.y_cord,
        )
        self.dist_white_to_target = math.hypot(
            target.x_cord - white.x_cord, target.y_cord - white.y_cord
        )

        self.score_angle = self.calculate_score_angle(self.angle)
        self.score_distance = self.calculate_score_distance(
            self.dist_white_to_target, self.dist_target_to_pocket
        )
        self.score = self.score_angle * self.score_distance

    def _set_as_invalid(self) -> None:
        """Helper method to set the state for an invalid shot."""
        self.pocket = None
        self.angle = float("inf")
        self.dist_target_to_pocket = float("inf")
        self.dist_white_to_target = float("inf")
        self.score_angle = -1.0
        self.score_distance = -1.0
        self.score = -1.0
        self.valid = False

    @staticmethod
    def calculate_score_angle(angle: float) -> float:
        """Calculates a score from 1 to 100 based on the shot angle."""
        abs_angle = abs(angle)
        if abs_angle >= 90:
            return 1.0
        return max(1.0, 100 * (1 - abs_angle / 90.0))

    @staticmethod
    def calculate_score_distance(
        dist_white_to_target: float,
        dist_target_to_pocket: float,
        weight_target_to_pocket: float = 0.7,
    ) -> float:
        """
        Calculates a score based on distances, with more weight on the
        target-to-pocket distance. Shorter distances yield higher scores.
        """
        norm_white = dist_white_to_target / get_max_white_to_target_distance()
        norm_target = dist_target_to_pocket / get_max_white_to_target_distance()

        weighted_avg = (
            1 - weight_target_to_pocket
        ) * norm_white + weight_target_to_pocket * norm_target
        score = 1 - weighted_avg
        return max(0.0, min(1.0, score))

    def get_lines(self) -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Gets the line segments representing the path of a direct shot.

        Returns:
            A list containing two lines: white->target and target->pocket.
            Returns an empty list if the shot is not valid.
        """
        if not self.valid or self.pocket is None:
            return []

        line_white_to_target = (
            (self.white.x_cord, self.white.y_cord),
            (self.target.x_cord, self.target.y_cord),
        )
        line_target_to_pocket = (
            (self.target.x_cord, self.target.y_cord),
            self.pocket.center,
        )
        return [line_white_to_target, line_target_to_pocket]

    def __repr__(self) -> str:
        """Returns a string representation of the BestShot object."""
        if not self.valid:
            return f"BestShot(INVALID: no free shot for target_id={self.target.id})"

        return (
            f"BestShot(target_id={self.target.id}, "
            f"pocket_id={self.pocket.id if self.pocket else 'N/A'}, "
            f"angle={self.angle:.2f}, score={self.score:.2f})"
        )
