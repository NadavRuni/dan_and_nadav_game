import math
from typing import List

from sympy import false

from game_class.C_table import Table
from const_numbers import *
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
            v2x = pocket.center[0] - self.target.x_cord
            v2y = pocket.center[1] - self.target.y_cord

            # dot & cross
            dot = v1x * v2x + v1y * v2y
            cross = v1x * v2y - v1y * v2x

            # זווית ברדיאנים
            angle_rad = math.atan2(cross, dot)

            # המרה למעלות
            angle_deg = math.degrees(angle_rad)
            dist_target_to_pocket = math.hypot(pocket.center[0] - self.target.x_cord, pocket.center[1] - self.target.y_cord)
            if not flag_to_wall:
                if self.have_free_shot(pocket):
                    angles[pocket.id] = [angle_deg, dist_target_to_pocket]
                else:
                    angles[pocket.id] = [NOT_FREE_SHOT, angle_deg, dist_target_to_pocket]
            else:
                angles[pocket.id] = [angle_deg, dist_target_to_pocket]

        print("[DEBUG] angles for target ball id: (", self.target.x_cord,',', self.target.y_cord,')')

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

        dx = ball.x_cord - pocket.center[0]
        dy = ball.y_cord - pocket.center[1]

        distance = math.hypot(dx, dy)
        angle_rad = math.atan2(dy, dx)  # angle of vector (ball → pocket)
        angle_deg = math.degrees(angle_rad)
        if (pocket.id == 0) and ball.id == 1:
            print(
                "for pocket id:",
                pocket.id,
                ", (",
                pocket.center[0],
                ",",
                pocket.center[1],
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
        dx = pocket.center[0] - target.x_cord
        dy = pocket.center[1] - target.y_cord
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
                if dist < ball.radius + target.radius + get_safe_distance():
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
            if len(values) == 2 and isinstance(values[0], (int, float)): # Now expects [angle, distance]
                angle_deg = values[0]
                dist_target_to_pocket = values[1]

                # Reuse BestShot's scoring logic for preliminary score
                # Import BestShot.calculate_score_angle and BestShot.calculate_score_distance here
                # Or re-implement similar logic
                from game_class.C_bestShot import BestShot # Import within function for locality

                # Placeholder for score_white_to_target (not available here, so set to a default)
                # We only have target_to_pocket distance readily available.
                # Let's assume white_to_target is roughly a fixed maximum for this preliminary score
                dist_white_to_target_placeholder = get_max_white_to_target_distance() / 2 

                score_angle = BestShot.calculate_score_angle(angle_deg)
                score_distance = BestShot.calculate_score_distance(
                    dist_white_to_target_placeholder,
                    dist_target_to_pocket
                )
                temp_score = score_angle * score_distance
                valid_angles[pid] = (temp_score, angle_deg) # Store score and original angle

        if not valid_angles:
            return NOT_FREE_SHOT, NOT_FREE_SHOT

        # Select the pocket with the highest temporary score
        best_pid = -1
        max_score = -1.0
        best_angle_for_pid = float("inf")

        for pid, (score, angle) in valid_angles.items():
            if score > max_score:
                max_score = score
                best_pid = pid
                best_angle_for_pid = angle
            elif score == max_score and abs(angle) < abs(best_angle_for_pid): # Tie-breaker: prefer smaller angle
                best_pid = pid
                best_angle_for_pid = angle
        
        return best_pid, best_angle_for_pid
