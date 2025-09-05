import math
from typing import List

from sympy import false

from game_class.C_table import Table
from const_numbers import NOT_FREE_SHOT
import math
from .C_ball import Ball
from .C_pocket import Pocket


class Calculations:
    def __init__(self, white: Ball, target: Ball, table: Table):
        self.white = white
        self.target = target
        self.table = table
        self.pockets = table.get_pockets()
        self.balls = table.get_balls()

    def angle_to_pockets(self, flag_to_wall: bool = False):
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

            if not flag_to_wall:
                if self.have_free_shot(pocket):
                    angles[pocket.id] = [angle_deg]
                else:
                    angles[pocket.id] = [NOT_FREE_SHOT, angle_deg]
            else:
                angles[pocket.id] = [angle_deg]

        print("[DEBUG]")
        print(angles)
        return angles

    def ball_to_pocket_info(self, ball: Ball, pocket: Pocket) -> dict:
        """
        Calculate distance and angle from a ball to a pocket.

        Args:
            ball (Ball): The ball object
            pocket (Pocket): The pocket object

        Returns:
            dict: {
                "distance": float,   # distance between centers
                "angle_rad": float,  # angle in radians (0 = right, counter-clockwise positive)
                "angle_deg": float   # angle in degrees
            }
        """

        dx = ball.x_cord - pocket.x_cord
        dy = ball.y_cord - pocket.y_cord

        distance = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)  # angle of vector (ball → pocket)
        angle_deg = math.degrees(angle_rad)
        if (pocket.id == 0) and ball.id == 1:
            print(
                "for pocket id:",
                pocket.id,
                ", (",
                pocket.x_cord,
                ",",
                pocket.y_cord,
                ")",
            )
            print("for ball id:", ball.id, ", (", ball.x_cord, ",", ball.y_cord, ")")
            print("distance : ", distance)
            print("angle_rad : ", angle_rad)

            print("angle_deg : ", angle_deg)

        return {"distance": distance, "angle_rad": angle_rad, "angle_deg": angle_deg}

    def have_free_shot(self, pocket: Pocket) -> bool:
        """
        בודק אם מהכדור המטרה אל חור מסוים יש מסלול פנוי (בלי כדורים שחוסמים).
        """
        # קו מטרה -> חור
        target = self.target
        dx = pocket.x_cord - target.x_cord
        dy = pocket.y_cord - target.y_cord
        dist_target_pocket = math.hypot(dx, dy)

        for ball in self.balls:
            if ball.id == target.id:  # לא בודקים את הכדור עצמו
                continue

            # וקטור מטרה -> כדור
            bx = ball.x_cord - target.x_cord
            by = ball.y_cord - target.y_cord

            # היטל של הכדור על הקו
            t = (bx * dx + by * dy) / (dist_target_pocket**2)

            # בודקים רק אם ההיטל נמצא בין המטרה לחור
            if 0 < t < 1:
                # הנקודה הכי קרובה על הקו
                closest_x = target.x_cord + t * dx
                closest_y = target.y_cord + t * dy

                # מרחק ממרכז הכדור לנקודה הכי קרובה
                dist = math.hypot(ball.x_cord - closest_x, ball.y_cord - closest_y)

                # אם הכדור נוגע בקו (כולל רדיוס שלו ושל המטרה) → חסימה
                if dist < ball.radius + target.radius:
                    return False

        return True

    def min_abs_angle(self) -> tuple[int, float]:
        """
        Returns the pocket with the smallest absolute angle.
        If there is no valid pocket → returns (NOT_FREE_SHOT, NOT_FREE_SHOT).
        Now works with angle_to_pockets() that returns:
            {pocket_id: [angle]} if free
            {pocket_id: [NOT_FREE_SHOT, angle]} if blocked
        """
        angles = self.angle_to_pockets()

        valid_angles = {}
        for pid, values in angles.items():
            if len(values) == 1 and isinstance(values[0], (int, float)):
                # Free shot case → take the angle
                valid_angles[pid] = values[0]

        if not valid_angles:
            return NOT_FREE_SHOT, NOT_FREE_SHOT

        return min(valid_angles.items(), key=lambda kv: abs(kv[1]))
