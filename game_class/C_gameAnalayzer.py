"""
The main analysis engine for finding the best shots in a game state.
"""

import math
from typing import List, Union

from const_numbers import FORSE_WALL_SHOT, NOT_FREE_SHOT, get_safe_distance
from game_class.C_bestShot import BestShot
from game_class.C_bestShotBallToBall import BestShotBallToBall
from game_class.C_bestShot_use_wall import BestWallShot
from game_class.C_calc_using_wall import CalculationsWithWall
from game_class.C_table import Table
from game_class.C_ball import GameBall


class GameAnalayzer:
    """
    Analyzes a given table state to find the best possible shots.
    """

    def __init__(self, table: Table):
        """
        Initializes the GameAnalayzer.

        Args:
            table: The Table object representing the current game state.
        """
        self.table = table

    def find_best_overall_shot(
        self, my_ball_type: str = "all"
    ) -> List[Union[BestShot, BestShotBallToBall, BestWallShot]]:
        """
        Calculates the top three best shots for the given ball type.

        This function first searches for direct shots. If none are found, it
        proceeds to look for more complex shots like ball-to-ball combinations
        and wall shots.

        Args:
            my_ball_type: The type of ball to target ('solid', 'striped', 'black', or 'all').

        Returns:
            A list of up to three of the best shot objects found, sorted by score.
        """
        table = self.table
        white_ball = next((b for b in table.get_balls() if b.type == "white"), None)
        if not white_ball:
            print("❌ White ball not found on the table.")
            return []

        all_shots: List[BestShot] = []
        target_balls = self._get_target_balls(my_ball_type)

        for ball in target_balls:
            if not self.has_clear_path(white_ball, ball, table.get_balls()):
                print(f"Path to ball {ball.id} is blocked.")
                continue

            shot = BestShot(white_ball, ball, table)
            if shot.valid and shot.score > 1:
                all_shots.append(shot)

        if all_shots:
            sorted_shots = sorted(all_shots, key=lambda s: s.score, reverse=True)
            return sorted_shots[:3]

        print("❌ No valid direct shots found. Looking for non-trivial shots.")
        return self._find_non_trivial_shots(my_ball_type)

    def _get_target_balls(self, my_ball_type: str) -> List[GameBall]:
        """Determines the list of balls to target based on the player's type."""
        if my_ball_type == "all":
            return [b for b in self.table.get_balls() if b.type != "white"]

        type_map = {
            "solid": self.table.get_solid_balls(),
            "striped": self.table.get_striped_balls(),
            "black": self.table.get_black_ball(),
        }
        target_balls = type_map.get(my_ball_type, [])

        # If the intended targets are gone, aim for the black ball.
        if not target_balls:
            return self.table.get_black_ball()

        return target_balls

    def has_clear_path(
        self, ball1: GameBall, ball2: GameBall, all_balls: List[GameBall]
    ) -> bool:
        """
        Checks if there is a clear path between two balls.

        The path is considered blocked if another ball is within a safe distance
        of the line segment connecting the edges of the two balls.

        Args:
            ball1: The starting ball.
            ball2: The ending ball.
            all_balls: A list of all balls on the table to check for obstruction.

        Returns:
            True if the path is clear, False otherwise.
        """
        ax, ay = ball1.x_cord, ball1.y_cord
        bx, by = ball2.x_cord, ball2.y_cord

        dx, dy = bx - ax, by - ay
        seg_len = math.hypot(dx, dy)
        if seg_len < 1e-6:
            return False  # Balls are in the same position

        # Unit vector for the line direction
        ux, uy = dx / seg_len, dy / seg_len

        # Define the line segment between the balls' perimeters
        axp = ax + ux * ball1.radius
        ayp = ay + uy * ball1.radius
        bxp = bx - ux * ball2.radius
        byp = by - uy * ball2.radius

        line_dx, line_dy = bxp - axp, byp - ayp
        line_len_sq = line_dx**2 + line_dy**2
        if line_len_sq < 1e-6:
            return True  # The balls are touching

        for other_ball in all_balls:
            if other_ball is ball1 or other_ball is ball2:
                continue

            # Project the center of the other ball onto the line segment
            vec_to_other_x, vec_to_other_y = (
                other_ball.x_cord - axp,
                other_ball.y_cord - ayp,
            )
            t = (vec_to_other_x * line_dx + vec_to_other_y * line_dy) / line_len_sq

            # If the projection is outside the segment, it can't be blocking
            if not (0 < t < 1):
                continue

            closest_x = axp + t * line_dx
            closest_y = ayp + t * line_dy

            dist_to_line = math.hypot(
                other_ball.x_cord - closest_x, other_ball.y_cord - closest_y
            )

            # If the distance is less than the sum of radii, the path is blocked
            if dist_to_line <= (other_ball.radius + get_safe_distance()):
                return False

        return True

    def _find_non_trivial_shots(
        self, my_ball_type: str
    ) -> List[Union[BestShotBallToBall, BestWallShot]]:
        """
        Finds the best combination and wall shots available.
        """
        print("Searching for combination and wall shots...")
        white_ball = next(b for b in self.table.get_balls() if b.type == "white")

        # 1. Find Ball-to-Ball shots
        b2b_shots = []
        target_balls = self._get_target_balls(my_ball_type)
        for helper_ball in target_balls:
            for target_ball_inner in target_balls:
                if helper_ball.id == target_ball_inner.id:
                    continue
                if not self.has_clear_path(
                    white_ball, helper_ball, self.table.get_balls()
                ):
                    continue

                shot = BestShotBallToBall(
                    white_ball, target_ball_inner, helper_ball, self.table
                )
                if shot.valid:
                    b2b_shots.append(shot)

        if b2b_shots and not FORSE_WALL_SHOT:
            return sorted(b2b_shots, key=lambda s: s.score, reverse=True)[:3]

        # 2. If no B2B shots, find Wall shots
        print("No valid combination shots found. Searching for wall shots...")
        wall_shots = []
        for ball in target_balls:
            calc = CalculationsWithWall(white_ball, ball, self.table)
            for pocket in self.table.get_pockets():
                wall_shot = BestWallShot(calc, pocket)
                if wall_shot.valid:
                    wall_shots.append(wall_shot)

        if wall_shots:
            return sorted(wall_shots, key=lambda s: s.score, reverse=True)[:3]

        print("❌ No valid non-trivial shots found either.")
        return []
