from typing import Optional
from game_class.C_ball import Ball
from game_class.C_pocket import Pocket
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_calcBallToBall import CalculationsBallToBall
import math
from const_numbers import *
from game_class.C_calc import Calculations


class BestShotBallToBall:
    def __init__(self, white: Ball, target: Ball, target_helper: Ball, table: Table):
        self.white = white
        self.target = target
        self.target_helper = target_helper
        self.table = table

        calc_helper_to_target_to_pocket = Calculations(target_helper, target, table)
        best_pocket_id, best_angle_from_helper_to_target = (
            calc_helper_to_target_to_pocket.min_abs_angle()
        )

        if (best_pocket_id, best_angle_from_helper_to_target) == NOT_FREE_SHOT or best_angle_from_helper_to_target >45:
            # לא קיים שוט חוקי
            self.no_valid_shot()
        else:
            # שמירת הכיס שנבחר
            self.pocket: Pocket = next(
                p for p in table.pockets if p.id == best_pocket_id
            )
            self.angle_from_helper_to_target: float = best_angle_from_helper_to_target


            if abs(self.angle_from_helper_to_target) > 50:
                self.no_valid_shot()
                return

            # חישוב מרחקים
            self.dist_target_to_pocket = math.hypot(
                self.pocket.center[0] - target.x_cord,
                self.pocket.center[1] - target.y_cord,
            )
            self.dist_helper_to_target = math.hypot(
                target.x_cord - target_helper.x_cord,
                target.y_cord - target_helper.y_cord,
            )

            self.score_angle = self.calculate_score_angle(
                self.angle_from_helper_to_target
            )
            self.score_distance = self.calculate_score_distance(
                self.dist_helper_to_target, self.dist_target_to_pocket
            )
            self.score = self.score_angle * self.score_distance
            self.valid = True

    def no_valid_shot(self):
        """מעדכן את מצב השוט לא חוקי"""
        # לא קיים שוט חוקי
        self.pocket: Pocket | None = None
        self.angle_from_helper_to_target: float = float("inf")
        self.dist_target_to_pocket = float("inf")
        self.dist_helper_to_target = float("inf")
        self.score_angle = -1
        self.score_distance = -1
        self.score = -1
        self.valid = False

    @staticmethod
    def calculate_score_angle(angle: float) -> float:
        """מחשב ציון מ־1 עד 100 לפי גודל הזווית"""
        abs_angle = abs(angle)
        if abs_angle >= 90:
            return 1
        return max(1, 100 * (1 - abs_angle / 90))

    @staticmethod
    def calculate_score_distance(
        dist_helper_to_target: float, dist_target_to_pocket: float
    ) -> float:
        norm_white = dist_helper_to_target / get_max_white_to_target_distance()
        norm_target = dist_target_to_pocket / get_max_white_to_target_distance()
        score = 1 - (norm_white + norm_target) / 2  # ממוצע נורמליזציות
        return max(0.0, min(1.0, score))
    
    def get_pocket(self) -> int | None:
        """מחזירה את ה־ID של הכיס שנבחר, או None אם אין שוט חוקי"""
        return self.pocket.id if self.valid else None

    def get_score(self) -> float | None:
        return self.score

    def get_pocket_and_angle(self) -> tuple[int | None, float]:
        """מחזירה tuple עם (pocket_id, angle) או (None, inf) אם אין שוט חוקי"""
        return (
            (self.pocket.id, self.angle_from_helper_to_target)
            if self.valid
            else (None, float("inf"))
        )

    def __repr__(self):
        if not self.valid:
            return f"BestShot(INVALID: no free shot for target_id={self.target.id})"

        return (
            f"BestShot("
            f"target_id={self.target.id}, "
            f"helper_id={self.target_helper.id}, "
            f"pocket_id={self.pocket.id}, "
            f"angle={self.angle_from_helper_to_target:.2f}, "
            f"dist_helper_target={self.dist_helper_to_target:.2f}, "
            f"dist_target_pocket={self.dist_target_to_pocket:.2f}, "
            f"score_angle={self.score_angle:.2f}, "
            f"score_distance={self.score_distance:.3f}, "
            f"final_score={self.score:.2f})"
        )

    def angle_white_helper_target_pocket(self) -> Optional[float]:
        """
        Calculates the sum of the absolute angle changes along the path
        from white ball -> helper ball -> target ball -> pocket.
        This represents the total "turn" required for the shot path.
        A straight line path will have a value of 0.
        """
        if not self.valid:
            return None

        # --- Calculate angle at helper ball ---
        w_x, w_y = self.white.x_cord, self.white.y_cord
        h_x, h_y = self.target_helper.x_cord, self.target_helper.y_cord
        t_x, t_y = self.target.x_cord, self.target.y_cord

        # Vectors for the turn at the helper ball
        v_wh = (h_x - w_x, h_y - w_y)
        v_ht = (t_x - h_x, t_y - h_y)

        # Helper to calculate signed angle in degrees
        def _calculate_angle_degrees(v1, v2):
            # Check for zero vectors
            if (v1[0] == 0 and v1[1] == 0) or (v2[0] == 0 and v2[1] == 0):
                return 0.0
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            det = v1[0] * v2[1] - v1[1] * v2[0]
            angle_rad = math.atan2(det, dot)
            return math.degrees(angle_rad)

        angle_at_helper = _calculate_angle_degrees(v_wh, v_ht)

        # --- Angle at target ball ---
        # This is already calculated in the constructor.
        angle_at_target = self.angle_from_helper_to_target

        # Sum of absolute angle changes to represent total "turn"
        return abs(angle_at_helper) + abs(angle_at_target)
