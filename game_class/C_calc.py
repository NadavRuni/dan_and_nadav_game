import math
from typing import List
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_table import Table
from const_numbers import NOT_FREE_SHOT



class Calculations:
    def __init__(self, white: Ball, target: Ball, table: Table ):
        self.white = white
        self.target = target
        self.pockets = table.get_pockets()
        self.balls = table.get_balls()


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

            if (self.have_free_shot(pocket)):
                angles[pocket.id] = angle_deg
            else: angles[pocket.id] = NOT_FREE_SHOT

        return angles

    def have_free_shot(self , pocket: Pocket) -> bool:
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
            t = (bx * dx + by * dy) / (dist_target_pocket ** 2)

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
        מחזירה את החור עם הזווית הקטנה ביותר בערך מוחלט.
        אם אין חור חוקי → מחזירה NOT_FREE_SHOT.
        """
        angles = self.angle_to_pockets()

        # סינון ערכים שהם מספרים בלבד
        valid_angles = {
            pid: ang for pid, ang in angles.items() if isinstance(ang, (int, float))
        }

        if not valid_angles:
            return NOT_FREE_SHOT , NOT_FREE_SHOT

        return min(valid_angles.items(), key=lambda kv: abs(kv[1]))



