"""
Represents and evaluates a ball-to-ball combination shot.

This module defines the BestShotBallToBall class, which analyzes the geometry
and feasibility of a shot where the cue ball hits a 'helper' ball, which in
turn hits the target ball into a pocket.
"""

import math
from typing import Optional

from const_numbers import (
    NOT_FREE_SHOT,
    get_max_white_to_target_distance,
)
from game_class.C_ball import GameBall
from game_class.C_calc import Calculations
from game_class.C_pocket import Pocket
from game_class.C_table import Table


class BestShotBallToBall:
    """
    Analyzes and scores a single ball-to-ball combination shot.

    The constructor for this class performs all the necessary calculations to
    determine if a valid path exists and, if so, scores the shot based on
    angles and distances.

    Attributes:
        white (GameBall): The cue ball.
        target (GameBall): The final target ball to be pocketed.
        target_helper (GameBall): The intermediate ball.
        table (Table): The table object containing all game state.
        pocket (Optional[Pocket]): The best pocket for the shot, if one exists.
        valid (bool): True if a valid shot was found, False otherwise.
        score (float): The final calculated score for the shot.
    """

    def __init__(
        self, white: GameBall, target: GameBall, target_helper: GameBall, table: Table
    ):
        """
        Initializes and evaluates the ball-to-ball shot.
        """
        self.white = white
        self.target = target
        self.target_helper = target_helper
        self.table = table
        self.valid = False

        # Find the best shot for the helper ball to pocket the target ball
        calc_helper_to_target = Calculations(target_helper, target, table)
        pocket_id, angle_helper_to_target = calc_helper_to_target.find_best_shot_angle()

        # Check if the helper-to-target shot is valid
        if (pocket_id, angle_helper_to_target) == NOT_FREE_SHOT:
            self._set_as_invalid()
            return

        # Check if the overall shot path has too sharp a turn
        total_turn_angle = self._calculate_total_turn_angle(angle_helper_to_target)
        if total_turn_angle is None or total_turn_angle > 45:
            self._set_as_invalid()
            return

        if abs(angle_helper_to_target) > 50:
            self._set_as_invalid()
            return

        # A valid shot path exists, so we can calculate the score
        self.valid = True
        self.pocket: Optional[Pocket] = next(
            (p for p in table.pockets if p.id == pocket_id), None
        )
        self.angle_from_helper_to_target: float = angle_helper_to_target

        self.dist_target_to_pocket = math.hypot(
            self.pocket.center[0] - target.x_cord,
            self.pocket.center[1] - target.y_cord,
        )
        self.dist_helper_to_target = math.hypot(
            target.x_cord - target_helper.x_cord,
            target.y_cord - target_helper.y_cord,
        )

        self.score_angle = self._calculate_angle_score(self.angle_from_helper_to_target)
        self.score_distance = self._calculate_distance_score(
            self.dist_helper_to_target, self.dist_target_to_pocket
        )
        self.score = self.score_angle * self.score_distance

    def _set_as_invalid(self):
        """Helper method to set the state for an invalid shot."""
        self.pocket = None
        self.angle_from_helper_to_target = float("inf")
        self.dist_target_to_pocket = float("inf")
        self.dist_helper_to_target = float("inf")
        self.score_angle = -1.0
        self.score_distance = -1.0
        self.score = -1.0
        self.valid = False

    @staticmethod
    def _calculate_angle_score(angle: float) -> float:
        """Calculates a score from 1 to 100 based on the shot angle."""
        abs_angle = abs(angle)
        if abs_angle >= 90:
            return 1.0
        return max(1.0, 100 * (1 - abs_angle / 90.0))

    @staticmethod
    def _calculate_distance_score(
        dist_helper_to_target: float, dist_target_to_pocket: float
    ) -> float:
        """Calculates a score based on the distances in the shot."""
        norm_helper_dist = dist_helper_to_target / get_max_white_to_target_distance()
        norm_target_dist = dist_target_to_pocket / get_max_white_to_target_distance()
        # Average the normalized distances
        score = 1 - (norm_helper_dist + norm_target_dist) / 2
        return max(0.0, min(1.0, score))

    def _calculate_total_turn_angle(self, angle_at_target: float) -> Optional[float]:
        """
        Calculates the sum of the absolute angle changes along the shot path.
        """
        w_x, w_y = self.white.x_cord, self.white.y_cord
        h_x, h_y = self.target_helper.x_cord, self.target_helper.y_cord
        t_x, t_y = self.target.x_cord, self.target.y_cord

        # Vectors for the turn at the helper ball
        vec_white_to_helper = (h_x - w_x, h_y - w_y)
        vec_helper_to_target = (t_x - h_x, t_y - h_y)

        def _calculate_signed_angle_degrees(v1, v2):
            if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
                return 0.0
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            return math.degrees(math.atan2(det, dot))

        angle_at_helper = _calculate_signed_angle_degrees(
            vec_white_to_helper, vec_helper_to_target
        )

        return abs(angle_at_helper) + abs(angle_at_target)

    def __repr__(self) -> str:
        """Returns a string representation of the shot."""
        if not self.valid:
            return f"BestShotBallToBall(INVALID: target_id={self.target.id})"

        return (
            f"BestShotBallToBall("
            f"target_id={self.target.id}, "
            f"helper_id={self.target_helper.id}, "
            f"pocket_id={self.pocket.id if self.pocket else 'N/A'}, "
            f"angle={self.angle_from_helper_to_target:.2f}, "
            f"score={self.score:.2f})"
        )
