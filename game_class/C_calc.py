import math
from typing import List
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket


class Calculations:
    def __init__(self, white: Ball, target: Ball, pockets: List[Pocket]):
        self.white = white
        self.target = target
        self.pockets = pockets

    def angle_to_pockets(self):
        """
        מחזיר מילון של {pocket_id: angle} שבו הזווית היא ההפרש
        בין הכיוון לבן→מטרה (נחשב כ-0°) לבין הכיוון מטרה→כיס.
        """
        angles = {}

        # וקטור לבן→מטרה
        v1x = self.target.x_cord - self.white.x_cord
        v1y = self.target.y_cord - self.white.y_cord

        for pocket in self.pockets:
            # וקטור מטרה→כיס
            v2x = pocket.x_cord - self.target.x_cord
            v2y = pocket.y_cord - self.target.y_cord

            # dot & cross
            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x

            # זווית ברדיאנים
            angle_rad = math.atan2(cross, dot)

            # המרה למעלות
            angle_deg = math.degrees(angle_rad)

            angles[pocket.id] = angle_deg

        return angles

    def min_abs_angle(self) -> tuple[int, float]:
        """
        מחזירה את החור עם הזווית הקטנה ביותר בערך מוחלט.
        Returns:
            (pocket_id, angle)
        """
        angles = self.angle_to_pockets()
        return min(angles.items(), key=lambda kv: abs(kv[1]))
