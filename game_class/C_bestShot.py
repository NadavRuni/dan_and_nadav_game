NOT_FREE_SHOT = (-1, float("inf"))
from game_class.C_ball import Ball
from game_class.C_table import Table
from game_class.C_calc import Calculations
from game_class.C_pocket import Pocket
import math
from const_numbers import *


class BestShot:
    def __init__(self, white: Ball, target: Ball, table: Table):
        self.white = white
        self.target = target
        self.table = table

        # חישוב כיס עם זווית מינימלית
        calc = Calculations(white, target, table)
        best_pocket_id, best_angle = calc.min_abs_angle()

        if best_pocket_id == NOT_FREE_SHOT or best_angle == float("inf"):
            # לא קיים שוט חוקי
            self.no_valid_shot()
        else:
            print(f"Best pocket: {best_pocket_id}, Best angle: {best_angle}")

            # שמירת הכיס שנבחר
            self.pocket: Pocket = next(
                p for p in table.pockets if p.id == best_pocket_id
            )
            self.angle: float = best_angle

            if abs(self.angle) > 85:
                self.no_valid_shot()
                return

            # חישוב מרחקים
            self.dist_target_to_pocket = math.hypot(
                self.pocket.x_cord - target.x_cord, self.pocket.y_cord - target.y_cord
            )
            self.dist_white_to_target = math.hypot(
                target.x_cord - white.x_cord, target.y_cord - white.y_cord
            )

            self.score_angle = self.calculate_score_angle(self.angle)
            self.score_distance = self.calculate_score_distance(
                self.dist_white_to_target, self.dist_target_to_pocket
            )
            self.score = self.score_angle * self.score_distance
            self.valid = True

    def no_valid_shot(self):
        """מעדכן את מצב השוט לא חוקי"""
        # לא קיים שוט חוקי
        self.pocket: Pocket | None = None
        self.angle: float = float("inf")
        self.dist_target_to_pocket = float("inf")
        self.dist_white_to_target = float("inf")
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

    def get_lines(self) -> list[tuple[tuple[float, float], tuple[float, float]]]:
        """
        Default lines for a direct shot:
          1. white → target
          2. target → pocket
        """
        if not self.valid or self.pocket is None:
            return []

        line_white_to_target = (
            (self.white.x_cord, self.white.y_cord),
            (self.target.x_cord, self.target.y_cord),
        )
        line_target_to_pocket = (
            (self.target.x_cord, self.target.y_cord),
            (self.pocket.x_cord, self.pocket.y_cord),
        )
        return [line_white_to_target, line_target_to_pocket]

    @staticmethod
    def calculate_score_distance(
        dist_white_to_target: float, dist_target_to_pocket: float
    ) -> float:
        norm_white = dist_white_to_target / MAX_WHITE_TO_TARGET
        norm_target = dist_target_to_pocket / MAX_TARGET_TO_POCKET
        score = 1 - (norm_white + norm_target) / 2  # ממוצע נורמליזציות
        return max(0.0, min(1.0, score))

    def get_pocket(self) -> int | None:
        """מחזירה את ה־ID של הכיס שנבחר, או None אם אין שוט חוקי"""
        return self.pocket.id if self.valid else None

    def get_score(self) -> float | None:
        return self.score

    def get_pocket_and_angle(self) -> tuple[int | None, float]:
        """מחזירה tuple עם (pocket_id, angle) או (None, inf) אם אין שוט חוקי"""
        return (self.pocket.id, self.angle) if self.valid else (None, float("inf"))

    def __repr__(self):
        if not self.valid:
            return f"BestShot(INVALID: no free shot for target_id={self.target.id})"

        return (
            f"BestShot("
            f"target_id={self.target.id}, "
            f"pocket_id={self.pocket.id}, "
            f"angle={self.angle:.2f}, "
            f"dist_white_target={self.dist_white_to_target:.2f}, "
            f"dist_target_pocket={self.dist_target_to_pocket:.2f}, "
            f"score_angle={self.score_angle:.2f}, "
            f"score_distance={self.score_distance:.3f}, "
            f"final_score={self.score:.2f})"
        )
