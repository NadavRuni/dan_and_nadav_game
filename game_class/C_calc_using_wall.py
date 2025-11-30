"""
Extends the base calculation class to handle shots involving a wall.

This module provides a class that calculates the geometry for shots that
require the cue ball to bounce off a wall before hitting the target.
"""

import math
from typing import Dict, Tuple

from const_numbers import NOT_FREE_SHOT, get_table_width
from game_class.C_ball import GameBall
from game_class.C_calc import Calculations
from game_class.C_pocket import Pocket
from game_class.C_table import Table


class CalculationsWithWall(Calculations):
    """
    Extends the base Calculations class to handle wall-shot geometry.
    """

    def __init__(self, white_ball: GameBall, target_ball: GameBall, table: Table):
        """
        Initializes the wall-shot calculation object.

        Args:
            white_ball: The cue ball.
            target_ball: The target ball.
            table: The table object containing game state.
        """
        super().__init__(white_ball, target_ball, table)
        self.distance_from_wall: Dict[str, float] = (
            self._calculate_distance_from_walls()
        )

    def _calculate_distance_from_walls(self) -> Dict[str, float]:
        """
        Calculates the distance from the target ball's center to each of the four walls.

        Returns:
            A dictionary with the distances to the 'left', 'right', 'up', and 'down' walls.
        """
        return {
            "left": self.target.x_cord,
            "right": self.table.length - self.target.x_cord,
            "down": self.target.y_cord,
            "up": self.table.width - self.target.y_cord,
        }

    def get_angles_to_pockets_via_wall(
        self,
    ) -> Dict[int, Tuple[float, Tuple[float, float]]]:
        """
        Calculates the angle and impact point for a wall shot to each corner pocket.

        Returns:
            A dictionary mapping pocket IDs to a tuple containing the required shot
            angle and the (x, y) impact point on the wall.
        """
        angle_using_wall_dict = {}
        for pocket in self.pockets:
            # Note: This logic currently only supports corner pockets.
            if pocket.id in {0, 1, 2, 3}:
                angle_using_wall_dict[pocket.id] = self._calculate_wall_shot_to_pocket(
                    pocket
                )
        return angle_using_wall_dict

    def _calculate_wall_shot_to_pocket(
        self, pocket: Pocket
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Calculates the required angle and impact point for a one-wall bank shot.

        This method uses the principle of reflection by creating a "virtual"
        pocket mirrored across the target wall. A straight line from the ball
        to the virtual pocket gives the correct angle.

        Args:
            pocket: The target pocket.

        Returns:
            A tuple containing:
            - The required shot angle in degrees.
            - The (x, y) coordinates of the impact point on the wall.
        """
        # Mirror the pocket across the top wall (y = table_width)
        mirrored_pocket_y = 2 * get_table_width() - pocket.center[1]

        # These represent the sides of a right triangle formed by the ball,
        # the mirrored pocket, and a line parallel to the wall.
        # This logic is complex and depends on the pocket's location.
        if pocket.id in {0, 1}:  # Bottom corner pockets via top wall
            q_side = mirrored_pocket_y - self.distance_from_wall["down"]
            side_wall_direction = "left" if pocket.id == 0 else "right"
        else:  # Top corner pockets via top wall
            q_side = self.distance_from_wall["down"] + get_table_width()
            side_wall_direction = "left" if pocket.id == 3 else "right"

        p_side = self.distance_from_wall[side_wall_direction]
        if p_side == 0:
            return 90.0, (self.target.x_cord, get_table_width())

        angle_rad = math.atan(q_side / p_side)
        angle_deg = abs(math.degrees(angle_rad))

        # Determine final angle and impact point based on pocket geometry
        impact_y = get_table_width()  # All shots here are via the top wall
        impact_x = self.target.x_cord + (
            q_side * self.distance_from_wall["up"] / p_side
        )

        if pocket.id == 0:  # Bottom-left
            impact_x = self.target.x_cord - self.distance_from_wall["up"] / math.tan(
                angle_rad
            )
            angle_deg = 180 - angle_deg
        elif pocket.id == 1:  # Bottom-right
            impact_x = self.target.x_cord + self.distance_from_wall["up"] / math.tan(
                angle_rad
            )
        elif pocket.id == 2:  # Top-right
            # This case seems to have inverted logic in the original code
            impact_x = self.target.x_cord + self.distance_from_wall["up"] / math.tan(
                angle_rad
            )
        elif pocket.id == 3:  # Top-left
            impact_x = self.target.x_cord - self.distance_from_wall["up"] / math.tan(
                angle_rad
            )
            angle_deg = 180 - angle_deg

        return angle_deg, (impact_x, impact_y)

    def find_best_wall_shot_angle(self) -> tuple:
        """
        Finds the corner pocket with the smallest absolute angle for a wall shot.

        Returns:
            A tuple of (pocket_id, angle) for the best shot, or
            (NOT_FREE_SHOT, NOT_FREE_SHOT) if no valid shot is found.
        """
        angles = self.get_angles_to_pockets_via_wall()

        # We only care about the angle, not the impact point for ranking
        valid_angles = {
            pid: angle_info[0]
            for pid, angle_info in angles.items()
            if angle_info is not None
        }

        if not valid_angles:
            return NOT_FREE_SHOT, NOT_FREE_SHOT

        # Return the pocket_id and angle (not impact point) of the best shot
        best_pocket_id, best_angle = min(
            valid_angles.items(), key=lambda kv: abs(kv[1])
        )
        return best_pocket_id, best_angle
