"""
Performs geometric calculations for finding and evaluating shots.

This module provides the Calculations class, which is used to determine angles,
distances, and clear paths between balls and pockets.
"""

import math
from typing import Dict, List, Tuple, Any

from const_numbers import (
    NOT_FREE_SHOT,
    get_safe_distance,
    get_max_white_to_target_distance,
)
from game_class.C_ball import GameBall
from game_class.C_pocket import Pocket
from game_class.C_table import Table


class Calculations:
    """
    A collection of methods for geometric calculations in the game.
    """

    def __init__(self, white_ball: GameBall, target_ball: GameBall, table: Table):
        """
        Initializes the Calculations object.

        Args:
            white_ball: The cue ball.
            target_ball: The target ball.
            table: The table object containing the game state.
        """
        self.white = white_ball
        self.target = target_ball
        self.table = table
        self.pockets = table.get_pockets()
        self.balls = table.get_balls()

    def get_angles_to_pockets(self) -> Dict[int, List[Any]]:
        """
        Calculates the angle from the cue ball's path to each pocket.

        For each pocket, this calculates the angle formed by the line from the
        cue ball to the target ball and the line from the target ball to the
        pocket. It also checks if the path is clear.

        Returns:
            A dictionary mapping pocket_id to a list. The list contains
            [angle, distance] if the shot is clear, or [NOT_FREE_SHOT, angle,
            distance] if it is blocked.
        """
        angles = {}
        # Vector from white ball to target ball
        v_white_target_x = self.target.x_cord - self.white.x_cord
        v_white_target_y = self.target.y_cord - self.white.y_cord

        for pocket in self.pockets:
            # Vector from target ball to pocket
            v_target_pocket_x = pocket.center[0] - self.target.x_cord
            v_target_pocket_y = pocket.center[1] - self.target.y_cord

            dot_product = (v_white_target_x * v_target_pocket_x) + (
                v_white_target_y * v_target_pocket_y
            )
            cross_product = (v_white_target_x * v_target_pocket_y) - (
                v_white_target_y * v_target_pocket_x
            )

            angle_rad = math.atan2(cross_product, dot_product)
            angle_deg = math.degrees(angle_rad)
            dist_target_to_pocket = math.hypot(v_target_pocket_x, v_target_pocket_y)

            if self.has_clear_path_to_pocket(pocket):
                angles[pocket.id] = [angle_deg, dist_target_to_pocket]
            else:
                angles[pocket.id] = [NOT_FREE_SHOT, angle_deg, dist_target_to_pocket]
        return angles

    def has_clear_path_to_pocket(self, pocket: Pocket) -> bool:
        """
        Checks if there is a clear path from the target ball to a pocket.

        Args:
            pocket: The pocket to check the path to.

        Returns:
            True if the path is clear of other balls, False otherwise.
        """
        target = self.target
        dx = pocket.center[0] - target.x_cord
        dy = pocket.center[1] - target.y_cord
        dist_target_pocket_sq = dx**2 + dy**2
        if dist_target_pocket_sq == 0:
            return True

        for ball in self.balls:
            # Don't check against the target ball itself
            if ball.id == target.id:
                continue

            # Project the ball's center onto the line from target to pocket
            vec_target_ball_x = ball.x_cord - target.x_cord
            vec_target_ball_y = ball.y_cord - target.y_cord

            projection = (
                vec_target_ball_x * dx + vec_target_ball_y * dy
            ) / dist_target_pocket_sq

            # Check if the projection falls between the target and the pocket
            if 0 < projection < 1:
                closest_x = target.x_cord + projection * dx
                closest_y = target.y_cord + projection * dy

                dist_to_line = math.hypot(
                    ball.x_cord - closest_x, ball.y_cord - closest_y
                )

                # Check for collision
                if dist_to_line < (ball.radius + target.radius + get_safe_distance()):
                    return False
        return True

    def find_best_shot_angle(self) -> Tuple[int, float]:
        """
        Finds the best shot by calculating a temporary score for each valid angle.

        Warning:
            This function imports `BestShot` locally, which is a code smell
            and can lead to circular dependencies.

        Returns:
            A tuple (pocket_id, angle) for the best shot, or (NOT_FREE_SHOT,
            NOT_FREE_SHOT) if no valid shot is found.
        """
        angles = self.get_angles_to_pockets()
        valid_angles = {}

        # This local import is problematic.
        from game_class.C_bestShot import BestShot

        for pid, values in angles.items():
            if len(values) == 2 and isinstance(values[0], (int, float)):
                angle_deg, dist_target_to_pocket = values

                # Use a placeholder for the white-to-target distance as it's
                # not directly available in this context.
                dist_white_to_target_placeholder = (
                    get_max_white_to_target_distance() / 2
                )

                score_angle = BestShot.calculate_score_angle(angle_deg)
                score_distance = BestShot.calculate_score_distance(
                    dist_white_to_target_placeholder, dist_target_to_pocket
                )
                temp_score = score_angle * score_distance
                valid_angles[pid] = (temp_score, angle_deg)

        if not valid_angles:
            return NOT_FREE_SHOT, NOT_FREE_SHOT

        # Select the pocket with the highest temporary score, using the
        # smallest angle as a tie-breaker.
        best_pid, (best_score, best_angle) = max(
            valid_angles.items(), key=lambda item: (item[1][0], -abs(item[1][1]))
        )

        return best_pid, best_angle
